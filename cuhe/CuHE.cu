/*
The MIT License (MIT)

Copyright (c) 2015 Wei Dai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "CuHE.h"
#include "Debug.h"
#include "Base.h"

namespace cuHE {

// Pre-computation
void CuHE::initRelin(ZZX* evalkey) {
	CSC(cudaSetDevice(0));	
	h_ek = new uint64* [param->numCrtPrime];
	for (int i=0; i<param->numCrtPrime; i++)
		CSC(cudaMallocHost(&h_ek[i], param->numEvalKey*param->nttLen*sizeof(uint64)));

	for (int i=0; i<param->numEvalKey; i++) {
		CuCtxt* ek = new CuCtxt(this);
		ek->setLevel(0, 0, evalkey[i]);
		ek->x2n();
		for (int j=0; j<param->numCrtPrime; j++)
			CSC(cudaMemcpy(h_ek[j]+i*param->nttLen, ek->nRep()+j*param->nttLen,
					param->nttLen*sizeof(uint64), cudaMemcpyDeviceToHost));

		ek->~CuCtxt();
	}

	d_relin = new uint64* [dm->numDevices()];
	d_ek = new uint64** [dm->numDevices()];
	dm->onAllDevices([=](int dev) {
		CSC(cudaSetDevice(dev));
		CSC(cudaMalloc(&d_relin[dev],
				param->numEvalKey*param->nttLen*sizeof(uint64)));
		d_ek[dev] = new uint64* [more];
		for (int i=0; i<more; i++)
			CSC(cudaMalloc(&d_ek[dev][i],
					param->numEvalKey*param->nttLen*sizeof(uint64)));
//		for (int i=0; i<more; i++)
//			CSC(cudaMemcpy(d_ek[dev]+i*param.numEvalKey*param.nttLen,
//						h_ek[dev*more+i], param.numEvalKey*param.nttLen*sizeof(uint64),
//						cudaMemcpyHostToDevice));
	});
}

// Operations
void CuHE::relinearization(uint64 *dst, uint32 *src, int lvl, int dev,
		cudaStream_t st) {
	CSC(cudaSetDevice(dev));
	op->nttw(d_relin[dev], src, param->_logCoeff(lvl), dev, st);
	for (int i=0; i<param->_numCrtPrime(lvl); i++) {
		CSC(cudaMemcpyAsync(d_ek[dev][i%more], h_ek[i],
				param->_numEvalKey(lvl)*param->nttLen*sizeof(uint64),
				cudaMemcpyHostToDevice, st));
		relinMulAddPerCrt<<<(param->nttLen+63)/64, 64, 0, st>>>(dst+i*param->nttLen,
				d_relin[dev], d_ek[dev][i%more], param->_numEvalKey(lvl), param->nttLen);
		CCE();
	}
}

CuHE::CuHE(GlobalParameters* _param, DeviceManager* _dm) {
	param = _param;
	dm = _dm;
	op = new Operations(param, dm);
}

CuHE::~CuHE() {
	delete op;
}

void CuHE::initCuHE(ZZ *coeffMod_, ZZX modulus) {
	dhBuffer_ = new uint32 *[dm->numDevices()];
	dm->onAllDevices([=](int i) {
		CSC(cudaSetDevice(i));
		CSC(cudaMallocHost(&dhBuffer_[i],
				param->rawLen*param->_wordsCoeff(0)*sizeof(uint32)));
		for (int j=0; j<dm->numDevices(); j++) {
			if (i != j)
				CSC(cudaDeviceEnablePeerAccess(j, 0));
		}
	});
	op->initNtt();
	initCrt(coeffMod_);
	initBarrett(modulus);
}

void CuHE::initCrt(ZZ* coeffModulus) {
	genCrtPrimes();
	op->genCoeffModuli();
	genCrtInvPrimes();
	op->genIcrt();
	dm->onAllDevices([=](int dev) {
		op->loadIcrtConst(0, dev);
	});
	op->getCoeffModuli(coeffModulus);
}

void CuHE::genCrtPrimes() {
	int pnum = param->numCrtPrime;
	op->crtPrime = new ZZ[pnum];
	unsigned* h_p = new unsigned[pnum];
	int logmid = param->logCoeffMin-(pnum-param->depth)*param->logCrtPrime;
	// after cutting, fairly larger primes
	ZZ temp = to_ZZ(0x1<<param->logCrtPrime)-1;
	for (int i=0; i<=pnum-param->depth-1; i++) {
		while (!ProbPrime(temp, 10))
			temp --;
		conv(h_p[i], temp);
		op->crtPrime[i] = temp;
		temp --;
	}

	// mid
	ZZ tmid;
	if (logmid != param->logCrtPrime)
		tmid = to_ZZ(0x1<<logmid)-1;
	else
		tmid = temp;
	while (!ProbPrime(tmid, 10))
		tmid --;
	conv(h_p[pnum-param->depth], tmid);
	op->crtPrime[pnum-param->depth] = tmid;

	// for cutting
	if (param->logCoeffCut == logmid)
		temp = tmid-1;
	else if (param->logCoeffCut == param->logCrtPrime)
		temp --;
	else
		temp = to_ZZ(0x1<<param->logCoeffCut)-1;
	for (int i=pnum-param->depth+1; i<pnum; i++) {
		while (!ProbPrime(temp, 10) || temp%to_ZZ(param->modMsg) != 1)
			temp --;
		conv(h_p[i], temp);
		op->crtPrime[i] = temp;
		temp --;
	}

	preload_crt_p(dm, h_p, pnum);
	delete [] h_p;
};


void CuHE::genCrtInvPrimes() {
	int pnum = param->numCrtPrime;
	uint32 *h_pinv = new uint32[pnum*(pnum-1)/2];
	ZZ temp;
	for (int i=1; i<pnum; i++)
		for (int j=0; j<i; j++)
			conv(h_pinv[i*(i-1)/2+j], InvMod(op->crtPrime[i]%op->crtPrime[j], op->crtPrime[j]));
	preload_crt_invp(dm, h_pinv, pnum*(pnum-1)/2);
	delete [] h_pinv;
}

void CuHE::setPolyModulus(ZZX m) {
	// compute NTL type zm, zu
	ZZ zq = op->coeffModulus[0];
	ZZX zm = m;
	ZZX zu;
	SetCoeff(zu, 2*param->modLen-1, 1);
	zu /= zm;
	for (int i=0; i<=deg(zm); i++)
		SetCoeff(zm, i, coeff(zm, i)%zq);
	for (int i=0; i<=deg(zu); i++)
		SetCoeff(zu, i, coeff(zu, i)%zq);
	SetCoeff(zm, param->modLen, 0);
	// prep m
	CuCtxt* c = new CuCtxt(this);
	c->setLevel(0, 0, zm);
	c->x2c();
	preload_barrett_m_c(dm, c->cRep(), param->numCrtPrime*param->crtLen*sizeof(uint32));
	c->x2n();
	preload_barrett_m_n(dm, c->nRep(), param->numCrtPrime*param->nttLen*sizeof(uint64));	
	// prep u
	CuCtxt* cc = new CuCtxt(this);
	cc->setLevel(0, 0, zu);
	cc->x2n();
	preload_barrett_u_n(dm, cc->nRep(),
			param->numCrtPrime*param->nttLen*sizeof(uint64));

	c->~CuCtxt();
	cc->~CuCtxt();
}

uint32 **CuHE::getDhBuffer() { return dhBuffer_; }

uint32 *CuHE::ptrBarrettCrt(int dev) { return d_barrett_crt[dev];}
uint64 *CuHE::ptrBarrettNtt(int dev) { return d_barrett_ntt[dev];}
uint32 *CuHE::ptrBarrettSrc(int dev) { return d_barrett_src[dev];}

void CuHE::createBarrettTemporySpace() {
	d_barrett_crt = new uint32*[dm->numDevices()];
	d_barrett_ntt = new uint64*[dm->numDevices()];
	d_barrett_src = new uint32*[dm->numDevices()];
	dm->onAllDevices([=](int dev) {
		cudaSetDevice(dev);
		CSC(cudaMalloc(&d_barrett_crt[dev],
				param->numCrtPrime*param->nttLen*sizeof(uint32)));
		CSC(cudaMalloc(&d_barrett_ntt[dev],
				param->numCrtPrime*param->nttLen*sizeof(uint64)));
		CSC(cudaMalloc(&d_barrett_src[dev],
				param->numCrtPrime*param->nttLen*sizeof(uint32)));
	});
}

void CuHE::initBarrett(ZZX m) {
	setPolyModulus(m);
	createBarrettTemporySpace();
}

void CuHE::startAllocator() {
	dm->bootDeviceAllocator(param->numCrtPrime*param->nttLen*sizeof(uint64));
}

void CuHE::stopAllocator() {
	dm->haltDeviceAllocator();
}

void CuHE::multiGPUs(int num) {
	dm->setNumDevices(num);
}

int CuHE::numGPUs() {
	return dm->numDevices();
}

void CuHE::setParameters(int d, int p, int w, int min, int cut, int m) {
	setParam(d, p, w, min, cut, m);
}

void CuHE::resetParameters() {
	resetParam(param);
}

void CuHE::initRelinearization(ZZX* evalkey) {
	initRelin(evalkey);
}

// Operations: CuCtxt & CuPtxt
void CuHE::copy(CuCtxt& dst, CuCtxt src, cudaStream_t st) {
	if (&dst != &src) {
		dst.reset();
		dst.setLevel(src.level(), src.domain(), src.device(), st);
		dst.isProd(src.isProd());
		CSC(cudaSetDevice(dst.device()));
		if (dst.domain() == 0)
			dst.zRep(src.zRep());
		else if (dst.domain() == 1)
			CSC(cudaMemcpyAsync(dst.rRep(), src.rRep(),
					dst.rRepSize(), cudaMemcpyDeviceToDevice, st));
		else if (dst.domain() == 2)
			CSC(cudaMemcpyAsync(dst.cRep(), src.cRep(),
					dst.cRepSize(), cudaMemcpyDeviceToDevice, st));
		else if (dst.domain() == 3)
			CSC(cudaMemcpyAsync(dst.nRep(), src.nRep(),
					dst.nRepSize(), cudaMemcpyDeviceToDevice, st));
		CSC(cudaStreamSynchronize(st));
	}
}
void CuHE::cAnd(CuCtxt& out, CuCtxt& in0, CuCtxt& in1, cudaStream_t st) {
	if (in0.device() != in1.device()) {
		cout<<"Error: Multiplication of different devices!"<<endl;
		terminate();
	}
	if (in0.domain() != 3 || in1.domain() != 3) {
		cout<<"Error: Multiplication of non-NTT domain!"<<endl;
		terminate();
	}
	if (in0.logq() != in1.logq()) {
		cout<<"Error: Multiplication of different levels!"<<endl;
		terminate();
	}
	if (&out != &in0) {
		out.reset();
		out.setLevel(in0.level(), 3, in0.device(), st);
	}
	CSC(cudaSetDevice(out.device()));
	op->nttMul(out.nRep(), in0.nRep(), in1.nRep(), out.logq(), out.device(), st);
	out.isProd(true);
	CSC(cudaStreamSynchronize(st));
}
void CuHE::cAnd(CuCtxt& out, CuCtxt& inc, CuPtxt& inp, cudaStream_t st) {
	if (inc.device() != inp.device()) {
		cout<<"Error: Multiplication of different devices!"<<endl;
		terminate();
	}
	if (inc.domain() != 3 || inp.domain() != 3) {
		cout<<"Error: Multiplication of non-NTT domain!"<<endl;
		terminate();
	}
	if (&out != &inc) {
		out.reset();
		out.setLevel(inc.level(), 3, inc.device(), st);
	}
	CSC(cudaSetDevice(out.device()));
	op->nttMulNX1(out.nRep(), inc.nRep(), inp.nRep(), out.logq(), out.device(), st);
	out.isProd(true);
	CSC(cudaStreamSynchronize(st));
}
void CuHE::cXor(CuCtxt& out, CuCtxt& in0, CuCtxt& in1, cudaStream_t st) {
	if (in0.device() != in1.device()) {
		cout<<"Error: Addition of different devices!"<<endl;
		terminate();
	}
	if (in0.logq() != in1.logq()) {
		cout<<"Error: Addition of different levels!"<<endl;
		terminate();
	}
	if (in0.domain() == 2 && in1.domain() == 2) {
		if (&out != &in0) {
			out.reset();
			out.setLevel(in0.level(), 2, in0.device(), st);
		}
		CSC(cudaSetDevice(out.device()));
		op->crtAdd(out.cRep(), in0.cRep(), in1.cRep(), out.logq(), out.device(), st);
		CSC(cudaStreamSynchronize(st));
	}
	else if (in0.domain() == 3 && in1.domain() == 3) {
		if (&out != &in0) {
			out.reset();
			out.setLevel(in0.level(), 3, in0.device(), st);
			out.isProd(in0.isProd()||in1.isProd());
		}
		CSC(cudaSetDevice(out.device()));
		op->nttAdd(out.nRep(), in0.nRep(), in1.nRep(), out.logq(), out.device(), st);
		CSC(cudaStreamSynchronize(st));
	}
	else {
		cout<<"Error: Addition of non-CRT-nor-NTT domain!"<<endl;
		terminate();
	}
}
void CuHE::cXor(CuCtxt& out, CuCtxt& in0, CuPtxt& in1, cudaStream_t st) {
	if (in0.device() != in1.device()) {
		cout<<"Error: Addition of different devices!"<<endl;
		terminate();
	}
	if (in0.domain() == 2 && in1.domain() == 2) {
		if (&out != &in0) {
			out.reset();
			out.setLevel(in0.level(), 2, in0.device(), st);
		}
		CSC(cudaSetDevice(out.device()));
		op->crtAddNX1(out.cRep(), in0.cRep(), in1.cRep(), out.logq(), out.device(), st);
		CSC(cudaStreamSynchronize(st));
	}
	else if (in0.domain() == 3 && in1.domain() == 3) {
		if (&out != &in0) {
			out.reset();
			out.setLevel(in0.level(), 3, in0.device(), st);
			out.isProd(in0.isProd()||in1.isProd());
		}
		CSC(cudaSetDevice(out.device()));
		op->nttAddNX1(out.nRep(), in0.nRep(), in1.nRep(), out.logq(), out.device(), st);
		CSC(cudaStreamSynchronize(st));
	}
	else {
		cout<<"Error: Addition of non-CRT-nor-NTT domain!"<<endl;
		terminate();
	}
}
void CuHE::cNot(CuCtxt& out, CuCtxt& in, cudaStream_t st) {
	if (in.domain() != 2) {
		cout<<"Error: cNot of non-CRT domain!"<<endl;
		terminate();
	}
	if (&out != &in) {
		out.reset();
		out.setLevel(in.level(), in.domain(), in.device(), st);
	}
	CSC(cudaSetDevice(out.device()));
	op->crtAddInt(out.cRep(), in.cRep(), (unsigned)param->modMsg-1, out.logq(),
			out.device(), st);
	CSC(cudaStreamSynchronize(st));
}
void CuHE::moveTo(CuCtxt& tar, int dstDev, cudaStream_t st) {
	if (dstDev != tar.device()) {
		void *ptr;
		if (tar.domain() == 1) {
			CSC(cudaSetDevice(dstDev));
			ptr = dm->deviceMalloc(tar.rRepSize());
			CSC(cudaSetDevice(tar.device()));
			CSC(cudaMemcpyPeerAsync(ptr, dstDev, tar.rRep(), tar.device(),
					tar.rRepSize(), st));
			tar.rRepFree();
			tar.rRep((uint32 *)ptr);
			CSC(cudaStreamSynchronize(st));
		}
		else if (tar.domain() == 2) {
			CSC(cudaSetDevice(dstDev));
			ptr = dm->deviceMalloc(tar.cRepSize());
			CSC(cudaSetDevice(tar.device()));
			CSC(cudaMemcpyPeerAsync(ptr, dstDev, tar.cRep(), tar.device(),
					tar.cRepSize(), st));
			tar.cRepFree();
			tar.cRep((uint32 *)ptr);
			CSC(cudaStreamSynchronize(st));
		}
		else if (tar.domain() == 3) {
			CSC(cudaSetDevice(dstDev));
			ptr = dm->deviceMalloc(tar.nRepSize());
			CSC(cudaSetDevice(tar.device()));
			CSC(cudaMemcpyPeerAsync(ptr, dstDev, tar.nRep(), tar.device(),
					tar.nRepSize(), st));
			tar.nRepFree();
			tar.nRep((uint64 *)ptr);
			CSC(cudaStreamSynchronize(st));
		}
		tar.device(dstDev);
	}
}
void CuHE::copyTo(CuCtxt& dst, CuCtxt& src, int dstDev, cudaStream_t st) {
	copy(dst, src, st);
	moveTo(dst, dstDev, st);
}

// NTL Interface
void CuHE::mulZZX(ZZX& out, ZZX in0, ZZX in1, int lvl, int dev, cudaStream_t st) {
	CuCtxt* cin0 = new CuCtxt(this);
	CuCtxt* cin1 = new CuCtxt(this);
	cin0->setLevel(lvl, dev, in0);
	cin1->setLevel(lvl, dev, in1);
	cin0->x2n(st);
	cin1->x2n(st);
	cAnd(*cin0, *cin0, *cin1, st);
	cin0->x2z(st);
	out = cin0->zRep();
	cin0->~CuCtxt();
	cin1->~CuCtxt();
}

// @class CuPolynomial
// Constructor
CuPolynomial::CuPolynomial(CuHE* _cuhe) {
	cuhe = _cuhe;
	logq_ = -1;
	domain_ = -1;
	device_ = -1;
	clear(zRep_);
	rRep_ = NULL;
	cRep_ = NULL;
	nRep_ = NULL;
	isProd_ = 0;
}
CuPolynomial::~CuPolynomial() {
	reset();
}
void CuPolynomial::reset() {
	clear(zRep_);
	if (rRep_ != NULL)
		rRepFree();
	if (cRep_ != NULL)
		cRepFree();
	if (nRep_ != NULL)
		nRepFree();
	isProd_ = 0;
	logq_ = -1;
	domain_ = -1;
	device_ = -1;
}
// Set Methods
void CuPolynomial::logq(int val) { logq_ = val;}
void CuPolynomial::domain(int val) { domain_ = val;}
void CuPolynomial::device(int val) { device_ = val;}
void CuPolynomial::isProd(bool val) { isProd_ = val;}
void CuPolynomial::zRep(ZZX val) { zRep_ = val;}
void CuPolynomial::rRep(uint32* val) { rRep_ = val;}
void CuPolynomial::cRep(uint32* val) { cRep_ = val;}
void CuPolynomial::nRep(uint64* val) { nRep_ = val;}
// Get Methods
int CuPolynomial::logq() { return logq_;}
int CuPolynomial::device() { return device_;}
int CuPolynomial::domain() { return domain_;}
bool CuPolynomial::isProd() { return isProd_;}
ZZX CuPolynomial::zRep() { return zRep_;}
uint32 * CuPolynomial::rRep() { return rRep_;}
uint32 * CuPolynomial::cRep() { return cRep_;}
uint64 * CuPolynomial::nRep() { return nRep_;}
// Domain Conversions
void CuPolynomial::z2r(cudaStream_t st) {
	if (domain_ != 0) {
		printf("Error: Not in domain ZZX!\n");
		terminate();
	}
	CSC(cudaSetDevice(device_));
	rRepCreate(st);
	uint32 **dhBuffer_ = cuhe->getDhBuffer();
	for(int i=0; i<cuhe->param->rawLen; i++)
		BytesFromZZ((uint8 *)(dhBuffer_[device_]+i*coeffWords()),
				coeff(zRep_, i), coeffWords()*sizeof(uint32));
	CSC(cudaMemcpyAsync(rRep_, dhBuffer_[device_], rRepSize(),
				cudaMemcpyHostToDevice, st));
	cudaStreamSynchronize(st);
	clear(zRep_);
	domain_ = 1;
}
void CuPolynomial::r2z(cudaStream_t st) {
	if (domain_ != 1) {
		printf("Error: Not in domain RAW!\n");
		terminate();
	}
	CSC(cudaSetDevice(device_));
	uint32 **dhBuffer_ = cuhe->getDhBuffer();
	CSC(cudaMemcpyAsync(dhBuffer_[device_], rRep_, rRepSize(),
			cudaMemcpyDeviceToHost, st));
	cudaStreamSynchronize(st);
	clear(zRep_);
	for(int i=0; i<cuhe->param->modLen; i++)
		SetCoeff( zRep_, i, ZZFromBytes( (uint8 *)(dhBuffer_[device_]
			+i*coeffWords() ), coeffWords()*sizeof(uint32)) );
	rRepFree();
	domain_ = 0;
}
void CuPolynomial::r2c(cudaStream_t st) {
	if (domain_ != 1) {
		printf("Error: Not in domain RAW!\n");
		terminate();
	}
	if (logq_ > cuhe->param->logCrtPrime) {
		CSC(cudaSetDevice(device_));
		cRepCreate(st);
		cuhe->op->crt(cRep_, rRep_, logq_, device_, st);
		rRepFree();
	}
	else {
		cRep_ = rRep_;
		rRep_ = NULL;
	}
	domain_ = 2;
}
void CuPolynomial::c2r(cudaStream_t st) {
	if (domain_ != 2) {
		printf("Error: Not in domain CRT!\n");
		terminate();
	}
	if (logq_ > cuhe->param->logCrtPrime) {
		CSC(cudaSetDevice(device_));
		rRepCreate(st);
		cuhe->op->icrt(rRep_, cRep_, logq_, device_, st);
		cRepFree();
	}
	else {
		rRep_ = cRep_;
		cRep_ = NULL;
	}
	domain_ = 1;
}
void CuPolynomial::c2n(cudaStream_t st) {
	if (domain_ != 2) {
		printf("Error: Not in domain CRT!\n");
		terminate();
	}
	CSC(cudaSetDevice(device_));
	nRepCreate(st);
	cuhe->op->ntt(nRep_, cRep_, logq_, device_, st);
	cRepFree();
	domain_ = 3;
}
void CuPolynomial::n2c(cudaStream_t st) {
	if (domain_ != 3) {
		printf("Error: Not in domain NTT!\n");
		terminate();
	}
	CSC(cudaSetDevice(device_));
	cRepCreate(st);
	if (isProd_) {
		cuhe->op->inttMod(cRep_, nRep_, logq_, device_, cuhe->ptrBarrettCrt(device_), cuhe->ptrBarrettNtt(device_), cuhe->ptrBarrettSrc(device_), st);
	}
	else {
		cuhe->op->intt(cRep_, nRep_, logq_, device_, st);
	}
	isProd_ = false;
	nRepFree();
	domain_ = 2;
}
void CuPolynomial::x2z(cudaStream_t st) {
	if (domain_ == 0)
		return;
	else if (domain_ == 1)
		r2z(st);
	else if (domain_ == 2) {
		c2r(st);
		r2z(st);
	}
	else if (domain_ == 3) {
		n2c(st);
		c2r(st);
		r2z(st);
	}
}
void CuPolynomial::x2r(cudaStream_t st) {
	if (domain_ == 1)
		return;
	else if (domain_ == 0)
		z2r(st);
	else if (domain_ == 2)
		c2r(st);
	else if (domain_ == 3) {
		n2c(st);
		c2r(st);
	}
}
void CuPolynomial::x2c(cudaStream_t st) {
	if (domain_ == 2)
		return;
	else if (domain_ == 0) {
		z2r(st);
		r2c(st);
	}
	else if (domain_ == 1)
		r2c(st);
	else if (domain_ == 3)
		n2c(st);
}
void CuPolynomial::x2n(cudaStream_t st) {
	if (domain_ == 3)
		return;
	else if (domain_ == 0) {
		z2r(st);
		r2c(st);
		c2n(st);
	}
	else if (domain_ == 1) {
		r2c(st);
		c2n(st);
	}
	else if (domain_ == 2)
		c2n(st);	
}
// Memory management
void CuPolynomial::rRepCreate(cudaStream_t st) {
	CSC(cudaSetDevice(device_));
	if (cuhe->dm->deviceAllocatorIsOn())
		rRep_ = (uint32 *)cuhe->dm->deviceMalloc(cuhe->param->numCrtPrime*cuhe->param->nttLen*sizeof(uint64));
	else
		CSC(cudaMalloc(&rRep_, rRepSize()));
	CSC(cudaMemsetAsync(rRep_, 0, rRepSize(), st));
}
void CuPolynomial::cRepCreate(cudaStream_t st) {
	CSC(cudaSetDevice(device_));
	if (cuhe->dm->deviceAllocatorIsOn())
		cRep_ = (uint32 *)cuhe->dm->deviceMalloc(cuhe->param->numCrtPrime*cuhe->param->nttLen*sizeof(uint64));
	else
		CSC(cudaMalloc(&cRep_, cRepSize()));
	CSC(cudaMemsetAsync(cRep_, 0, cRepSize(), st));
}
void CuPolynomial::nRepCreate(cudaStream_t st) {
	CSC(cudaSetDevice(device_));
	if (cuhe->dm->deviceAllocatorIsOn())
		nRep_ = (uint64 *)cuhe->dm->deviceMalloc(cuhe->param->numCrtPrime*cuhe->param->nttLen*sizeof(uint64));
	else
		CSC(cudaMalloc(&nRep_, nRepSize()));
	CSC(cudaMemsetAsync(nRep_, 0, nRepSize(), st));
}
void CuPolynomial::rRepFree() {
	CSC(cudaSetDevice(device_));
	if (cuhe->dm->deviceAllocatorIsOn())
		cuhe->dm->deviceFree(rRep_);
	else
		CSC(cudaFree(rRep_));
	rRep_ = NULL;
}
void CuPolynomial::cRepFree() {
	CSC(cudaSetDevice(device_));
	if (cuhe->dm->deviceAllocatorIsOn())
		cuhe->dm->deviceFree(cRep_);
	else
		CSC(cudaFree(cRep_));
	cRep_ = NULL;
}
void CuPolynomial::nRepFree() {
	CSC(cudaSetDevice(device_));
	if (cuhe->dm->deviceAllocatorIsOn())
		cuhe->dm->deviceFree(nRep_);
	else
		CSC(cudaFree(nRep_));
	nRep_ = NULL;
}
// Utilities
int CuPolynomial::coeffWords() { return (logq_+31)/32;}
size_t CuPolynomial::rRepSize() { return cuhe->param->rawLen*coeffWords()*sizeof(uint32);}

// @class CuCtxt
// Get Methods
void CuCtxt::setLevel(int lvl, int domain, int device, cudaStream_t st) {
	level_ = lvl;
	logq_ = cuhe->param->_logCoeff(lvl);
	domain_ = domain;
	device_ = device;
	if (domain_ == 0)
		clear (zRep_);
	else if (domain_ == 1)
		rRepCreate(st);
	else if (domain_ == 2)
		cRepCreate(st);
	else if (domain_ == 3)
		nRepCreate(st);
}
void CuCtxt::setLevel(int lvl, int device, ZZX val) {
	level_ = lvl;
	logq_ = cuhe->param->_logCoeff(lvl);
	domain_ = 0;
	device_ = device;
	zRep_ = val;
}
int CuCtxt::level() { return level_;}
// Noise Control
void CuCtxt::modSwitch(cudaStream_t st) {
	if (logq_ < cuhe->param->logCoeffMin+cuhe->param->logCoeffCut) {
		printf("Error: Cannot do modSwitch on last level!\n");
		terminate();
	}
	x2c();
	CSC(cudaSetDevice(device_));
	cuhe->op->crtModSwitch(cRep_, cRep_, logq_, device_, st);
	CSC(cudaStreamSynchronize(st));
	logq_ -= cuhe->param->logCoeffCut;
	level_ ++;
}
void CuCtxt::modSwitch(int lvl, cudaStream_t st) {
	if (lvl < level_ || lvl >= cuhe->param->depth) {
		printf("Error: ModSwitch to unavailable level!\n");
		terminate();
	}
	else if (lvl == level_)
		return;
	x2c();
	CSC(cudaSetDevice(device_));
	while (lvl > level_) {
		cuhe->op->crtModSwitch(cRep_, cRep_, logq_, device_, st);
		logq_ -= cuhe->param->logCoeffCut;
	}
	CSC(cudaStreamSynchronize(st));
}
void CuCtxt::relin(cudaStream_t st) {
	CSC(cudaSetDevice(device_));
	x2r();
	nRepCreate(st);
	cuhe->relinearization(nRep_, rRep_, level_, device_, st);
	CSC(cudaStreamSynchronize(st));
	rRepFree();
	isProd_ = true;
	domain_ = 3;
	n2c();
	CSC(cudaStreamSynchronize(st));
}
size_t CuCtxt::cRepSize() { return cuhe->param->_numCrtPrime(level_)*cuhe->param->crtLen*sizeof(uint32);}
size_t CuCtxt::nRepSize() { return cuhe->param->_numCrtPrime(level_)*cuhe->param->nttLen*sizeof(uint64);}

// @class CuPtxt
void CuPtxt::setLogq(int logq, int domain, int device, cudaStream_t st) {
	logq_ = logq;
	domain_ = domain;
	device_ = device;
	if (domain_ == 0)
		clear (zRep_);
	else if (domain_ == 1)
		rRepCreate(st);
	else if (domain_ == 2)
		cRepCreate(st);
	else if (domain_ == 3)
		nRepCreate(st);
}
void CuPtxt::setLogq(int logq, int device, ZZX val) {
	logq_ = logq;
	domain_ = 0;
	device_ = device;
	zRep_ = val;
}
size_t CuPtxt::cRepSize() { return cuhe->param->crtLen*sizeof(uint32);}
size_t CuPtxt::nRepSize() { return cuhe->param->nttLen*sizeof(uint64);}

} // namespace cuHE
