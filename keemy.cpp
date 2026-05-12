/*	keemy.cpp

        KentuckY Efficent Error Modeler

        This program denoises images using texture synthesis to literally
        create a new image with more consistent statistics based
        on a probability density function for pixel value errors.
        The PDF is computed from the image being processed by
        examining neighboring pixel values; there is no training set.

        2026 by Aaron Weitekamp and H. Dietz
*/

#include <iostream>
#include <math.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <filesystem>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <omp.h>

// --- CONSTANTS ---

#define EMDIM 256 // number of gray levels in the error model

#ifndef COLORS
#define COLORS 3 // default number of color channels
#endif

#if COLORS != 1 && COLORS != 3
#error COLORS must be 1 or 3
#endif

#define PDFTERMS (6 * 3) // number of PDF terms used in similarity metric
#define MINW (1.0/128)  // minimum weight for the self-patch 

#define POFFS 8 // number of neighbors used for histogram
int poff[POFFS];

// max number of patches stored for denoising per-pixel
// in-practice K = K_MAX
constexpr int K_MAX = 40; // 40 is a good sweet spot, but higher K can lead to better quality denoising

constexpr int PATCH_SIZE = 3;
constexpr int LEVELS_MAX = 10;  // safety cap on pyramid depth
constexpr int PYRAMID_MIN_DIM = 64;  // coarsest pyramid level minimum dimension

typedef uint8_t pixel_t;

char *myname;
char *infile;

 // --- STRUCTURES ---

struct DenoiserState {
  alignas(64) cv::Mat orgMat;
  alignas(64) cv::Mat padMat; // reflective padding
  alignas(64) cv::Mat resultMat;

  int xdim, ydim, dim;
  int stride;

  uint8_t K;

  alignas(64) pixel_t *org  = nullptr;
  alignas(64) pixel_t *porg = nullptr;

  // error model
  alignas(64) uint32_t hist[COLORS][EMDIM][EMDIM];
  alignas(64) float    pdf [COLORS][EMDIM][EMDIM];
  alignas(64) uint8_t  pdf8[EMDIM][EMDIM][COLORS];

  static constexpr int PATCH_RAD = (PATCH_SIZE - 1) / 2;
  static constexpr int MPOFFS = PATCH_SIZE * PATCH_SIZE - 1;

  int matchPoff[MPOFFS]; // offsets for patch extraction

  void initPatchOffs() {
    int idx = 0;
    for (int dy = -PATCH_RAD; dy <= PATCH_RAD; dy++)
      for (int dx = -PATCH_RAD; dx <= PATCH_RAD; dx++) {
        if (dx == 0 && dy == 0) continue;
        matchPoff[idx++] = dy * stride + dx * COLORS;
      }
  }

  DenoiserState(const cv::Mat &input)
      : orgMat(input.clone()), padMat(), resultMat(input.clone()),
        org(orgMat.data), porg(nullptr),
        xdim(orgMat.cols), ydim(orgMat.rows), dim(xdim * ydim) {

    // init padded img
    cv::copyMakeBorder(orgMat, padMat,
                       PATCH_RAD, PATCH_RAD, PATCH_RAD, PATCH_RAD,
                       cv::BORDER_REFLECT);

    porg   = padMat.data;
    stride = padMat.step;

    initPatchOffs();
  }

  ~DenoiserState() {
    orgMat.release();
    padMat.release();
    resultMat.release();
    org  = nullptr;
    porg = nullptr;
  }
};

// stores top-k matches
struct PatchData {

  // patch index offsets and similarity 
  alignas(64) int16_t dxs [K_MAX];
  alignas(64) int16_t dys [K_MAX];
  alignas(64) float sims[K_MAX];

  int count = 0; // number of matches stored so far
  int K = K_MAX;

  // bestIdx is queried by other patches
  // worstIdx is only used internally to speed up linear search
  int bestIdx = -1;  // index of max sim
  int worstIdx = -1; // index of min sim

  // fills up until K matches are found, then replaces worst match 
  void tryInsert(int dx, int dy, float sim) {
    
    // puts in match list if it's not full
    if (count < K) {
      dxs [count] = dx;
      dys [count] = dy;
      sims[count] = sim;

      if (bestIdx  == -1 || sim > sims[bestIdx])  bestIdx  = count;
      if (worstIdx == -1 || sim < sims[worstIdx]) worstIdx = count;

      ++count;
      return;
    }

    // when top-k full, check against worst similarity
    if (sim > sims[worstIdx]) {
      // replace the worst match
      dxs [worstIdx] = dx;
      dys [worstIdx] = dy;
      sims[worstIdx] = sim;

      // recompute bestIdx and worstIdx after replacement
      bestIdx = 0;
      worstIdx = 0;
      for (int i = 1; i < K; ++i) {
        if (sims[i] > sims[bestIdx])
          bestIdx = i;
        if (sims[i] < sims[worstIdx])
          worstIdx = i;
      }
    }
  }

  // retrieve best match
  int   getBestDx()  const { return dxs [bestIdx]; }
  int   getBestDy()  const { return dys [bestIdx]; }
  float getBestSim() const { return sims[bestIdx]; }
};


// --- HELPERS ---


// helper for indexing pixels from padded image
inline int padIdx(int x, int y, int stride) { return y * stride + x * COLORS; }

// thread-local random number generator
inline std::mt19937 &get_thread_local_gen() {
  static thread_local std::mt19937 gen([] {
    std::random_device rd;
    auto seed = rd() ^ (static_cast<std::mt19937::result_type>(
                            std::hash<std::thread::id>()(std::this_thread::get_id())) << 16);
    return seed;
  }());
  return gen;
}

inline int randint(int min, int max) {
  return std::uniform_int_distribution<>(min, max)(get_thread_local_gen());
}

// ensures random match offsets stay in image bounds
int randdx(int x, int xrad, int xdim) {
  return randint(-std::min(x, xrad), std::min(xdim - x - 1, xrad));
}
int randdy(int y, int yrad, int ydim) {
  return randint(-std::min(y, yrad), std::min(ydim - y - 1, yrad));
}

int computeCoarseThresh(int levels) {
  return std::max(1, levels * 2 / 3);
}

// determines number of pyramid levels
int computeLevelIters(int level, int levels) {
  if (levels <= 1) return 8;

  float t = (float)level / (float)(levels - 1);
  int base = (int)(4.0f + 8.0f * t);

  return std::max(3, std::min(base, 12));
}


// --- ERROR MODEL ---


void mkhist(DenoiserState &state) {
  
  // initialize hist; mostly 0, 1 on diagonal
  for (int c = 0; c < COLORS; ++c) {
    for (int j = 0; j < EMDIM; ++j) {
      for (int i = 0; i < EMDIM; ++i) {
        state.hist[c][j][i] = (i == j);
      }
    }
  }

  // everyone evals each pixel's neighbors,
  // but only marks itself (if appropriate)
  pixel_t *p = state.org + COLORS + COLORS * state.xdim;
  for (int row = 1; row < state.ydim - 1; ++row) {
    for (int col = 1; col < state.xdim - 1; ++col) {
      for (int c = 0; c < COLORS; ++c) {
        int v = *p;

        // find most similar neighbor of same color
        int near = p[poff[0]];
        int dif = abs(near - v);

        for (int k = 1; k < POFFS; ++k) {
          int n = p[poff[k]];
          int d = abs(n - v);
          if (d < dif) {
            dif = d;
            near = n;
          }
        }
        ++p;

        // bump everything between near and v
        if (near > v) {
          int t = near;
          near = v;
          v = t;
        }
        for (int j = near; j <= v; ++j) {
          for (int i = near; i <= v; ++i) {
            ++(state.hist[c][j][i]);
          }
        }
      }
    }
    p += (COLORS + COLORS);
  }
}

inline static void mkpdf(DenoiserState &state, int myx, int myy) {
  for (int c = 0; c < COLORS; ++c) {

    // make myv monotonic and compute max in my row
    int myv = state.hist[c][myy][myx]; // [color][y][x]
    int max = myv;
    for (int i = 0; i < EMDIM; ++i) {

      int v = state.hist[c][myy][i]; // [color][y][x]
      if (v > myv) {
        if (v > max)
          max = v;

        if (((myx > myy) && (i > myx)) || ((myx < myy) && (i < myx))) {
          myv = v;
        }
      }
    }

    // normalize probability density function (PDF)
    float f = myv / ((float)max);
    state.pdf[c][myy][myx] = powf(f, (1.0 / PDFTERMS)); // [color][y][x]
    state.pdf8[myy][myx][c] = 0.5 + (255.0 * f);        // [y][x][color]
  }
}

  // compute pixel error model from image
void mkerrmodel(DenoiserState &state, bool writeImage = false) {

  // patch offsets
  poff[0] = -COLORS - COLORS * state.xdim;
  poff[1] =  0      - COLORS * state.xdim;
  poff[2] =  COLORS - COLORS * state.xdim;
  poff[3] = -COLORS;
  poff[4] =  COLORS;
  poff[5] = -COLORS + COLORS * state.xdim;
  poff[6] =  0      + COLORS * state.xdim;
  poff[7] =  COLORS + COLORS * state.xdim;

  mkhist(state);

 // create probability density function from histogram
#pragma omp parallel for schedule(guided)
  for (int y = 0; y < EMDIM; ++y)
    for (int x = 0; x < EMDIM; ++x)
      mkpdf(state, x, y);

  if (writeImage) {
    cv::Mat pdfMat(EMDIM, EMDIM, CV_8UC3);
    memcpy(pdfMat.data, &state.pdf8[0][0][0], sizeof(state.pdf8));
    imwrite("errmod.png", pdfMat);
  }
}

template<int CH>
inline static float similarity(const DenoiserState &state, pixel_t *a, pixel_t *b) {

  float sim = state.pdf[0][a[0]][b[0]];
  for (int c = 1; c < CH; ++c) {
    sim *= state.pdf[c][a[c]][b[c]];
  }

  return sim;
}

inline static float simpatch(const DenoiserState &state, pixel_t *a, pixel_t *b) {
  const int MPOFFS = state.MPOFFS;

  // compute similarity of patches around a and b
  float sim[MPOFFS + 1];

  for (int i = 0; i < MPOFFS; ++i) {
    sim[i] = similarity<COLORS>(state, a + state.matchPoff[i], b + state.matchPoff[i]);
  }
  sim[MPOFFS] = sim[0];

  float best1 = sim[0]; // most similar pixel pair
  float best = sim[0] * sim[1]; // most similar pair of adjacent pixel pairs
  for (int i = 0; i < MPOFFS; ++i) {
    float s = sim[i] * sim[i + 1];

    if (s > best)
      best = s;

    if (sim[i] > best1)
      best1 = sim[i];
  }


  float s = similarity<COLORS>(state, a, b); // patch center similarity
  return (best1 * best1 * best * s * s);
}


// --- PATCH MATCH ---


// random inital patch match offsets and similarity
void initPatchMatch(DenoiserState &state, PatchData matches[],
                    int xrad = -1, int yrad = -1) {
  if (xrad == -1) xrad = state.xdim >> 1;
  if (yrad == -1) yrad = state.ydim >> 1;

  const int xdim = state.xdim;
  const int ydim = state.ydim;
  const int PATCH_RAD = state.PATCH_RAD;

#pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < ydim; ++y) {
    for (int x = 0; x < xdim; ++x) {
      int matchIdx = y * xdim + x;
      pixel_t *src = state.porg + padIdx(x + PATCH_RAD, y + PATCH_RAD, state.stride);

      // initialize with K random matches (excluding self)
      for (int k = 0; k < state.K; ++k) {
        int dx = randdx(x, xrad, xdim);
        int dy = randdy(y, yrad, ydim);

        while (dx == 0 && dy == 0) {
          dx = randdx(x, xrad, xdim);
          dy = randdy(y, yrad, ydim);
        }
        pixel_t *match = state.porg + padIdx(x + dx + PATCH_RAD,
                                              y + dy + PATCH_RAD, state.stride);
        matches[matchIdx].tryInsert(dx, dy, simpatch(state, src, match));
      }
    }
  }
}

// propagates best match to neighbor pixels
void propagate(DenoiserState &state, PatchData matches[],
               pixel_t *src, int matchIdx, int neighborIdx, int x, int y) {
  const int dx = matches[neighborIdx].getBestDx();
  const int dy = matches[neighborIdx].getBestDy();

  const int xdim = state.xdim;
  const int ydim = state.ydim;

  const int PATCH_RAD = state.PATCH_RAD;
  const int STRIDE = state.stride;

  // skip if offset self-patch
  if (dx == 0 && dy == 0) return;

  // bounds check: (x + dx, y + dy) must be in image bounds
  if (x + dx < 0 || x + dx >= xdim || y + dy < 0 || y + dy >= ydim) return;

  // propagates if better than worst match in patch stack
  pixel_t *match = state.porg + padIdx(x + dx + PATCH_RAD, y + dy + PATCH_RAD, STRIDE);
  matches[matchIdx].tryInsert(dx, dy, simpatch(state, src, match));
}

// patchmatch without enrichment and random search shrunk between iteration, not per-pixel search.
// parallelized using wavefront approach, i.e. iteration done over anti-diagonals
void patchMatch(DenoiserState &state, PatchData matches[], int iter,
                int &xrad, int &yrad, float *prevBestSims,
                const float IMPROVING_FRAC = 0.01f) {
  const int ydim = state.ydim;
  const int xdim = state.xdim;

  const int PATCH_RAD = state.PATCH_RAD;
  const int STRIDE = state.stride;


  int stillImproving = 0;

  const bool reverse = (iter % 2) == 1;
  if (!reverse) {
#pragma omp parallel reduction(+:stillImproving)
    {
      for (int s = 0; s < xdim + ydim - 1; ++s) {
#pragma omp for schedule(static) nowait
        for (int x = std::max(0, s - ydim + 1); x <= std::min(s, xdim - 1); ++x) {
          int y = s - x;
          const int matchIdx = y * xdim + x;
          pixel_t *src = state.porg + padIdx(x + PATCH_RAD, y + PATCH_RAD, STRIDE);

          if (x > 0) propagate(state, matches, src, matchIdx, y * xdim + (x-1), x, y);
          if (y > 0) propagate(state, matches, src, matchIdx, (y-1) * xdim + x, x, y);

          const int rdx = randdx(x, xrad, xdim);
          const int rdy = randdy(y, yrad, ydim);
          if (!(rdx == 0 && rdy == 0)) {
            pixel_t *rm = state.porg + padIdx(x + rdx + PATCH_RAD,
                                               y + rdy + PATCH_RAD, STRIDE);
            matches[matchIdx].tryInsert(rdx, rdy, simpatch(state, src, rm));
          }

          if (matches[matchIdx].count > 0 && xrad >= 4) {
            float newBest = matches[matchIdx].getBestSim();
            if (newBest > prevBestSims[matchIdx] + 1e-5f) ++stillImproving;
            prevBestSims[matchIdx] = newBest;
          }
        }
      }
    }
  } else {
#pragma omp parallel reduction(+:stillImproving)
    {
      for (int s = 0; s < xdim + ydim - 1; ++s) {
#pragma omp for schedule(static) nowait
        for (int x = xdim - 1 - std::min(s, xdim - 1);
                 x <= xdim - 1 - std::max(0, s - ydim + 1); ++x) {
          int y = ydim - 1 - (s - (xdim - 1 - x));
          const int matchIdx = y * xdim + x;
          pixel_t *src = state.porg + padIdx(x + PATCH_RAD, y + PATCH_RAD, STRIDE);

          if (x < xdim - 1) propagate(state, matches, src, matchIdx, y * xdim + (x+1), x, y);
          if (y < ydim - 1) propagate(state, matches, src, matchIdx, (y+1) * xdim + x, x, y);

          const int rdx = randdx(x, xrad, xdim);
          const int rdy = randdy(y, yrad, ydim);
          if (!(rdx == 0 && rdy == 0)) {
            pixel_t *rm = state.porg + padIdx(x + rdx + PATCH_RAD,
                                               y + rdy + PATCH_RAD, STRIDE);
            matches[matchIdx].tryInsert(rdx, rdy, simpatch(state, src, rm));
          }

          if (matches[matchIdx].count > 0 && xrad >= 4) {
            float newBest = matches[matchIdx].getBestSim();
            if (newBest > prevBestSims[matchIdx] + 1e-5f) ++stillImproving;
            prevBestSims[matchIdx] = newBest;
          }
        }
      }
    }
  }

  // early convergence check
  if (iter >= 1 && xrad >= 4) {
    float frac = (float)stillImproving / (float)(xdim * ydim);
    fprintf(stdout, "\t  improving: %.2f%%\n", frac * 100.0f);

    if (frac < IMPROVING_FRAC) {
      fprintf(stdout, "\tConverged: <%.1f%% pixels still improving\n",
              IMPROVING_FRAC * 100.0f);

      // setting to zero stops current pyramid search
      xrad = 0;
      yrad = 0;
    }
  }
}

// patchmatch algorithm wrapper, finds best matches/textures
void patchSearch(DenoiserState &state, PatchData matches[], int iters,
               bool coarse = false, int levelIdx = 0, int totalLevels = 1) {
  int xrad, yrad;
  if (coarse) {
    xrad = state.xdim >> 1;
    yrad = state.ydim >> 1;
  } 
  else {
    float t = (totalLevels > 1) ? (float)levelIdx / (float)(totalLevels - 1) : 0.0f;

    xrad = std::max(4, (int)(state.xdim * (0.05f + 0.15f * t)));
    yrad = std::max(4, (int)(state.ydim * (0.05f + 0.15f * t)));
  }

  // store previous best similarlity for convergence check
  std::vector<float> prevBestSims(state.dim, 0.0f);

  for (int i = 0; i < iters; ++i) {
    fprintf(stdout, "\titer: %d  (xrad=%d yrad=%d)\n", i, xrad, yrad);
    patchMatch(state, matches, i, xrad, yrad, prevBestSims.data());

    if (xrad < 4 && yrad < 4) break;

    xrad = std::max(4, xrad >> 1);
    yrad = std::max(4, yrad >> 1);
  }
}


// --- IMAGE PYRAMID / DENOISING ---


// maps patch stacks to pixels in the next higher resolution
// also recomputes similarity on higher resolution, as image content has changed
void upsampleMatches(DenoiserState &fineState, PatchData *fineMatches,
                         const DenoiserState &coarseState, const PatchData *coarseMatches) {
  const int fineW = fineState.xdim;
  const int fineH = fineState.ydim;

  const int coarseW = coarseState.xdim;
  const int coarseH = coarseState.ydim;

  const int scale = 2;

#pragma omp parallel for
  for (int fy = 0; fy < fineH; ++fy) {
    for (int fx = 0; fx < fineW; ++fx) {
      const int fineIdx = fy * fineW + fx;
      PatchData &fp = fineMatches[fineIdx];

      // get corresponding coarse pixel coordinates
      const int cx = fx / scale;
      const int cy = fy / scale;

      // out-of-bounds check
      if (cx >= coarseW || cy >= coarseH) continue;

      // get fine/current level 
      const PatchData &cp = coarseMatches[cy * coarseW + cx];
      pixel_t *fineSrc = fineState.porg + padIdx(fx + fineState.PATCH_RAD,
                                                     fy + fineState.PATCH_RAD,
                                                     fineState.stride);

      // update patch stack and similarity
      for (int k = 0; k < cp.count; ++k) {

        // random jitter added for robustness
        int fdx = scale * cp.dxs[k] + randint(-1, 1);
        int fdy = scale * cp.dys[k] + randint(-1, 1);
        int fmx = fx + fdx, fmy = fy + fdy;

        if (fmx < 0 || fmx >= fineW || fmy < 0 || fmy >= fineH ||
            (fdx == 0 && fdy == 0))
          continue;

        pixel_t *fm = fineState.porg + padIdx(fmx + fineState.PATCH_RAD,
                                               fmy + fineState.PATCH_RAD, fineState.stride);
        fp.tryInsert(fdx, fdy, simpatch(fineState, fineSrc, fm));
      }
    }
  }
}

// denoises using weighted average of patch centers
// alternatively, prunes patch stack tail using threshold to save finer detail
void denoise(DenoiserState &state, PatchData *matches, float simContributionThresh = 0.00f) {
    pixel_t *__restrict result = state.resultMat.ptr<pixel_t>();
    const pixel_t *__restrict orig  = state.org;
    const pixel_t *__restrict porg  = state.porg;

    const int ydim = state.ydim;
    const int xdim = state.xdim;

    const int PATCH_RAD = state.PATCH_RAD;
    const int STRIDE = state.stride;

    const bool CULL_K = (simContributionThresh > 0.0f);

#pragma omp parallel for schedule(static) collapse(2)
    for (int y = 0; y < ydim; ++y) {
        for (int x = 0; x < xdim; ++x) {
            const int idx    = y * xdim + x;
            PatchData &patch = matches[idx];

            const int count  = patch.count;

            const int inBase = idx * COLORS;
            const pixel_t *src = orig + inBase;

            // insertion sort descending by similarity, fast given low K
            if (CULL_K) {
              for (int i = 1; i < count; ++i) {
                  float ks = patch.sims[i];

                  int16_t kdx = patch.dxs[i];
                  int16_t kdy = patch.dys[i];

                  int j = i - 1;
                  while (j >= 0 && patch.sims[j] < ks) {
                      patch.sims[j+1] = patch.sims[j];
                      patch.dxs[j+1]  = patch.dxs[j];
                      patch.dys[j+1]  = patch.dys[j];

                      --j;
                  }
                  patch.sims[j+1] = ks;
                  patch.dxs[j+1]  = kdx;
                  patch.dys[j+1]  = kdy;
              }
            }

            // Accumulate matches in similarity order, stopping when the
            // marginal contribution drops below simContributionThresh * current sum.
            // In flat regions: similarities stay high, runs to full K.
            // In detail regions: similarities drop off, stops early.
            float qsum = MINW;
            float vsum[COLORS];
            for (int c = 0; c < COLORS; ++c)
                vsum[c] = qsum * src[c];

            for (int k = 0; k < count; ++k) {
                const float w = patch.sims[k];

                if (CULL_K && w < simContributionThresh * qsum)
                    break;

                const int matchBase = padIdx(x + patch.dxs[k] + PATCH_RAD,
                                             y + patch.dys[k] + PATCH_RAD, STRIDE);

                const pixel_t *match = porg + matchBase;
                for (int c = 0; c < COLORS; ++c)
                    vsum[c] += w * match[c];
                qsum += w;
            }

            const float invQ = 1.0f / qsum;
            for (int c = 0; c < COLORS; ++c)
                result[inBase + c] = vsum[c] * invQ;
        }
    }
}

// rescore updates matches that are stale from image content changing
// (regions that are similar remain similar, but by how much changes)
// after every two denoise passes patch stack rescored
void rescoreMatches(DenoiserState &state, PatchData *matches) {
  cv::Mat cleanPadded;
  cv::copyMakeBorder(state.resultMat, cleanPadded,
                     state.PATCH_RAD, state.PATCH_RAD,
                     state.PATCH_RAD, state.PATCH_RAD,
                     cv::BORDER_REFLECT);

  pixel_t  *cleanPorg = cleanPadded.data;
  const int stride = (int)cleanPadded.step;

  DenoiserState cleanState(state.resultMat);
  mkerrmodel(cleanState);

  const int xdim = state.xdim;
  const int ydim = state.ydim;
  const int PATCH_RAD = state.PATCH_RAD;

#pragma omp parallel for schedule(static) collapse(2)
  for (int y = 0; y < ydim; ++y) {
    for (int x = 0; x < xdim; ++x) {
      PatchData &patch = matches[y * xdim + x];
      pixel_t *src = cleanPorg + padIdx(x + PATCH_RAD, y + PATCH_RAD, stride);

      for (int k = 0; k < patch.count; ++k) {
        int mx = x + patch.dxs[k];
        int my = y + patch.dys[k];

        if (mx < 0 || mx >= xdim || my < 0 || my >= ydim) {
          patch.sims[k] = 0.0f;
          continue;
        }

        pixel_t *match = cleanPorg + padIdx(mx + PATCH_RAD, my + PATCH_RAD, stride);
        patch.sims[k] = simpatch(cleanState, src, match);
      }
    }
  }
}

// denoises image using image pyramid
void pyramidDenoise(DenoiserState &state, int levels = 3, int denoisePasses = 2, float simContributionThresh = 0.00f) {
  const int COARSE_THRESH = computeCoarseThresh(levels);

  fprintf(stdout, "Pyramid: %d levels, coarse threshold at level %d\n",
          levels, COARSE_THRESH);

  // denoises without pyramid, this won't work great as patchmatch implementation optimized for pyramid
  if (levels <= 1) {
    state.K = K_MAX;
    auto matches = std::make_unique<PatchData[]>(state.dim);

    constexpr bool writeImage = true;
    mkerrmodel(state, writeImage);

    initPatchMatch(state, matches.get());

    constexpr int noPyramidIters = 12;
    constexpr bool coarse = true;
    constexpr int levelIdx = 0;
    constexpr int totalLevels = 1;
    patchSearch(state, matches.get(), noPyramidIters, coarse, levelIdx, totalLevels);

    fprintf(stdout, "\tPass 1...\n");
    denoise(state, matches.get(), simContributionThresh);

    for (int pass = 1; pass < denoisePasses; ++pass) {
      if (pass % 2 == 0) {
        fprintf(stdout, "\tRescoring Matches...\n");
        rescoreMatches(state, matches.get());
      }

      fprintf(stdout, "\tPass %d...\n", pass + 1);

      state.orgMat = state.resultMat.clone();
      state.org    = state.orgMat.data;
      cv::copyMakeBorder(state.orgMat, state.padMat,
                        state.PATCH_RAD, state.PATCH_RAD,
                        state.PATCH_RAD, state.PATCH_RAD, cv::BORDER_REFLECT);
      state.porg = state.padMat.data;

      denoise(state, matches.get(), simContributionThresh);
    }

    matches.reset();
    return;
  }

  DenoiserState *pyramid[LEVELS_MAX];
  std::unique_ptr<PatchData[]>  matches[LEVELS_MAX];
  pyramid[0] = &state;

  // area averaged pyramid instead of gaussian as it preserves detail better
  // also seems to result in less color artifacts
  for (int i = 1; i < levels; ++i) {
    const DenoiserState &prev = *pyramid[i - 1];

    cv::Mat down;
    cv::resize(prev.orgMat, down,
               cv::Size(prev.orgMat.cols / 2, prev.orgMat.rows / 2),
               0, 0, cv::INTER_AREA);

    pyramid[i] = new DenoiserState(down);
  }

  // runs patchmatch from lowest resolution to highest
  for (int level = levels - 1; level >= 0; --level) {
    fprintf(stdout, "LEVEL: %d\n", level);

    DenoiserState &curr = *pyramid[level];
    curr.K = K_MAX;

    bool writeImage = (level == 0);
    mkerrmodel(curr, writeImage);

    matches[level] = std::make_unique<PatchData[]>(curr.dim);

    if (level == levels - 1) {
      fprintf(stdout, "\tPatch Match Init...\n");
      initPatchMatch(curr, matches[level].get());
    } 
    else {
      fprintf(stdout, "\tUpsampling Matches...\n");
      upsampleMatches(curr, matches[level].get(),
                      *pyramid[level + 1], matches[level + 1].get());

      matches[level + 1].reset();
      if (level + 1 > 0) {
        delete pyramid[level + 1];
        pyramid[level + 1] = nullptr;
      }
    }

    bool coarse = (level >= COARSE_THRESH);
    int levelIters = computeLevelIters(level, levels);
    fprintf(stdout, "\tIters for level %d: %d\n", level, levelIters);

    patchSearch(curr, matches[level].get(), levelIters, coarse, level, levels);
  }

  // denoises using found patches
  fprintf(stdout, "\nDenoising (%d passes):\n", denoisePasses);

  fprintf(stdout, "\tPass 1...\n");
  denoise(state, matches[0].get(), simContributionThresh);

  for (int pass = 1; pass < denoisePasses; ++pass) {
    if (pass % 2 == 0) {
      fprintf(stdout, "\tRescoring Matches...\n");
      rescoreMatches(state, matches[0].get());
    }

    fprintf(stdout, "\tPass %d...\n", pass + 1);

    state.orgMat = state.resultMat.clone();
    state.org    = state.orgMat.data;
    cv::copyMakeBorder(state.orgMat, state.padMat,
                       state.PATCH_RAD, state.PATCH_RAD,
                       state.PATCH_RAD, state.PATCH_RAD, cv::BORDER_REFLECT);
    state.porg = state.padMat.data;

    denoise(state, matches[0].get(), simContributionThresh);
  }

  matches[0].reset();
}

// determines number of pyramid levels from image resolution
int computePyramidLevels(int width, int height) {
  int levels = 0, w = width, h = height;
  while (w >= PYRAMID_MIN_DIM && h >= PYRAMID_MIN_DIM && levels < LEVELS_MAX) {
    w >>= 1; h >>= 1; ++levels;
  }

  return std::max(1, levels);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "Usage: %s <input> [output] [passes] [simThresh]\n",
            argv[0]);
    return 1;
  }

  // optional parameter defaults
  const char* outfile = "denoised.png";
  int denoisePasses = 2;
  float simContributionThresh = 0.00f;

  myname = argv[0];
  infile = argv[1];

  if (argc > 2)
    outfile = argv[2];

  if (argc > 3)
    denoisePasses = atoi(argv[3]);

  if (argc > 4)
    simContributionThresh = atof(argv[4]);

#if COLORS == 1
  cv::Mat orgMat = cv::imread(infile, cv::IMREAD_GRAYSCALE);
#else
  cv::Mat orgMat = cv::imread(infile, cv::IMREAD_COLOR);
#endif

  if (orgMat.empty()) {
    fprintf(stderr, "Failed to load image: %s\n", infile);
    return 1;
  }

  DenoiserState state(orgMat);

  int levels = computePyramidLevels(state.xdim, state.ydim);
  fprintf(stdout, "Image: %dx%d  ->  %d pyramid levels\n",
          state.xdim, state.ydim, levels);

  fprintf(stdout,
          "passes=%d  simThresh=%f  outfile=%s\n",
          denoisePasses,
          simContributionThresh,
          outfile);

  fprintf(stdout, "\n");

  pyramidDenoise(state,
                 levels,
                 denoisePasses,
                 simContributionThresh);

  imwrite(outfile, state.resultMat);

  return 0;
}
