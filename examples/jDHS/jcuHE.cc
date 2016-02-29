#include "eu_unicredit_cuHE_jcuHE.h"
#include <stdio.h>
#include <stdlib.h>

#include "../DHS/DHS.h"
#include "../../cuhe/CuHE.h"
using namespace cuHE;

jlong JNICALL Java_eu_unicredit_cuHE_jcuHE_encrypt
  (JNIEnv *env, jobject obj, jboolean value) {
  printf("called encrypt!\n");
  fflush(stdout);
  return (jlong)0L;
}

int main(void) {
	printf("Ciao!\n");
}
