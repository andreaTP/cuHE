#################################################
#################################################
## Compilers
CC	= gcc
CX	= g++
CU	= /usr/local/cuda/bin/nvcc

## Flags
FMACH	= -m64
FOPTM	= -O2
FC++11	= -std=c++11
FOMP	= 
# CUDA Flags
FARCH	= -arch=compute_30
FCODE	= -code=sm_30
FPTX	= -code=compute_30
CXFLAGS	= $(FMACH) $(FOPTM) $(FC++11) $(FOMP)
CUFLAGS	= $(FARCH) $(FCODE) $(FPTX)

## Libraries
LCUDART	= --cudart static
LMATH	= -lm
LNTL	= -lntl
LGMP	= -lgmp
LOMP	= -lomp
LINKERS	= $(LCUDART) $(LMATH) $(LNTL) $(LGMP) $(LOMP)

# Commands
COMPCC	= $(CC) $(CXFLAGS)
COMPCX	= $(CX) $(CXFLAGS)
COMPCU	= $(CU) $(CXFLAGS) $(CUFLAGS)
LINKCC	= $(CC) $(CXFLAGS) $(LINKERS) -link -w
LINKCX	= $(CX) $(CXFLAGS) $(LINKERS) -link -w
LINKCU	= $(CU) $(CXFLAGS) $(CUFLAGS) $(LINKERS) -link -w

#################################################
#################################################
## Source Files
SRCCC	=
SRCCX	=
SRCCU	= ../DHS/DHS.cu\
		  Prince.cu\
		  Timer.cu\
		  test_Prince.cu
SRCS	= $(SRCCC) $(SRCCX) $(SRCCU)

## Dependency Files
DEPCC	= $(SRCCC:.c=.d)
DEPCX	= $(SRCCX:.cpp=.d)
DEPCU	= $(SRCCU:.cu=.d)
DEPS	= $(DEPCC) $(DEPCX) $(DEPCU)

## Object Files
OBJCC	= $(SRCCC:.c=.o)
OBJCX	= $(SRCCX:.cpp=.o)
OBJCU	= $(SRCCU:.cu=.o)
OBJS	= $(OBJCC) $(OBJCX) $(OBJCU)

## Library Files
LIBS	= ../../cuhe/libcuHE.a

## Executable Targets
TARS	= test_Prince

#################################################
#################################################
all: $(TARS)

test_Prince: $(OBJS) $(LIBS)
	$(LINKCU) -o $@ $(OBJS) $(LIBS)

$(OBJCC): $(@:%.o=%.c)
	$(COMPCC) -M $(@:%.o=%.c) -o $(@:%.o=%.d)
	$(COMPCC) -c $(@:%.o=%.c) -o $@
$(OBJCX): $(@:%.o=%.cpp)
	$(COMPCX) -M $(@:%.o=%.cpp) -o $(@:%.o=%.d)
	$(COMPCX) -c $(@:%.o=%.cpp) -o $@
$(OBJCU): $(@:%.o=%.cu)
	$(COMPCU) -M $(@:%.o=%.cu) -o $(@:%.o=%.d)
	$(COMPCU) -c $(@:%.o=%.cu) -o $@

clean:
	rm $(DEPS) $(OBJS) $(TARS)