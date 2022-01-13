
## Path variables.
INCLUDES = LeNet-5\\
BUILD = Build\\
BIN = LeNet-5\\x64\\Debug\\
DEBUG = x64\\Debug\\
EXECUTABLE = $(BIN)LeNet-5.exe
SOURCE = LeNet-5\\
CC = nvcc

## Compile command variables.
CFLAGS = -I$(INCLUDES)
LIBS =

## Builds an optimized release build of the project.
debugcuda:
   $(CC) $(SOURCE)main.cu $(SOURCE)lenet.cu $(CFLAGS) $(LIBS) -g -D"DEBUG" -o NNet.exe
   @echo Build complete!

debugseq:
   $(CC) $(SOURCE)main.c $(SOURCE)lenet.c -g -D"DEBUG"
   @echo Build complete!

## Cleans the project folder.
clean:
   -del $(BIN)*.ilk *.lib *.exe *.exp *.obj *.pdb
   @echo Clean complete!