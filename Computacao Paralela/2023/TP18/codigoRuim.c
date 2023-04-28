/*Erros
codigoRuim.c:15:21: missed: not vectorized: control flow in loop.
/usr/include/x86_64-linux-gnu/bits/stdio2.h:112:10: missed: statement clobbers memory: __builtin_putchar (10);
codigoRuim.c:19:14: missed: not vectorized, possible dependence between data-refs c[_32] and c[j_43]
*/

#include <stdio.h>

#define N 1000

int main() {
  int a[N], b[N], c[N];

  // Preenche os vetores a e b com valores; isso tem que ficar separado senão vai dar erro de execução
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = N - i;
  }

  // Realiza operações complicadas com os valores de a e b
  for (int i = 1; i < N; i += 2) {
    int j = i + 1;
    int k = i * 2;

    c[i - 1] = a[i - 1] * b[i - 1];
    c[j] = b[j] * k;

    if (a[i] % 2 == 0) {
      c[i] = a[i] / b[i];
    } else {
      c[i] = a[i] + b[i] + 1;
    }

    if (b[j] > 0) {
      c[k] = a[k] - b[k] / 2;
    } else {
      c[k] = a[k] + b[k] * 2;
    }
  }

  // Imprime o resultado
  for (int i = 0; i < N; i++) {
    printf("%d ", c[i]);
  }
  printf("\n");

  return 0;
}
