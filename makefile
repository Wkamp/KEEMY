CXX = g++
CXXFLAGS = -O3
OPENCV = `pkg-config --cflags --libs opencv4`

SRC = keemy.cpp

all: keemy keemy_gray

keemy: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o keemy $(OPENCV)

keemy_gray: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -DCOLORS=1 -o keemy_gray $(OPENCV)

clean:
	rm -f keemy keemy_gray
