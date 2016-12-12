FROM nvidia/cuda:8.0-devel

MAINTAINER Andrea Peruffo <andrea.peruffo1982@gmail.com>

RUN apt-get -y update

RUN apt-get -y install software-properties-common

RUN add-apt-repository ppa:ubuntu-toolchain-r/test

# --- --- ---  Update OS --- --- ---
RUN apt-get -y update

# --- --- ---  Tar --- --- ---
RUN apt-get -y install tar bzip2

RUN apt-get -y install curl

# --- --- ---  Gcc / C++ --- --- ---
RUN apt-get -y install gcc-5 g++-5

# --- --- ---  Make --- --- ---
RUN apt-get -y install make

# --- --- ---  Gcc compilation deps --- --- ---
RUN apt-get -y install libgmp-dev
RUN apt-get -y install libmpfr-dev
RUN apt-get -y install libmpc-dev

#gmp gmp-devel mpfr mpfr-devel libmpc libmpc-devel

# --- --- ---  Libtool --- --- ---
RUN apt-get -y install libtool

WORKDIR /home

# --- --- ---  GCC 5.4 --- --- ---
#RUN curl -L "ftp://ftp.gnu.org/pub/gnu/gcc/gcc-5.4.0/gcc-5.4.0.tar.bz2" -o /home/gcc.tar.bz2
#RUN curl -L "http://gcc.skazkaforyou.com/releases/gcc-5.4.0/gcc-5.4.0.tar.bz2" -o /home/gcc.tar.bz2

#RUN tar xvfj /home/gcc.tar.bz2

#WORKDIR /home/gcc-5.4.0/build

#RUN chmod a+x ../configure

#RUN ../configure --disable-multilib --enable-languages=c,c++
#RUN make
#RUN make install

# --- --- ---  NTL --- --- ---
WORKDIR /home

RUN curl -L "http://www.shoup.net/ntl/ntl-10.3.0.tar.gz" -o /home/ntl.tar.gz

RUN tar xvf /home/ntl.tar.gz

WORKDIR /home/ntl-10.3.0/src

RUN ./configure SHARED=on NTL_EXCEPTIONS=on
RUN make
RUN make install


#Update GCC
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 60 --slave /usr/bin/g++ g++ /usr/bin/g++-5

# --- --- ---  Cmake --- --- ---
RUN apt install -y cmake

# Fix cuda GCC
#RUN ln -fs /usr/local/bin/gcc /usr/bin/gcc
#RUN ln -fs /usr/local/bin/g++ /usr/bin/g++
#RUN ln -fs /usr/local/bin/cpp /usr/bin/cpp
#RUN ln -fs /usr/local/bin/c++ /usr/bin/c++

#Fix library path
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib64/:/usr/local/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc

WORKDIR /home/sources

CMD ["/bin/bash"]
