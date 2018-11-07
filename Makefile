EXECUTABLE := bins

CC_FILES   := bin.cpp main.cpp

###########################################################

OBJDIR=objs

CXX=g++ -m64
CXXFLAGS=-O3 -Wall -Werror -std=c++11
LDFLAGS=
OBJS=$(OBJDIR)/bin.o $(OBJDIR)/main.o

default: $(EXECUTABLE)

clean:
	    rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

dirs:
	mkdir -p $(OBJDIR)/


$(EXECUTABLE): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	 $(CXX) $< $(CXXFLAGS) -c -o $@
