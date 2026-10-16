/* Phase 4 harness.
 * Build (guards.c is #included below because its functions are static inline; do NOT also compile it,
 * and do NOT compile dispatch.c):
 *   gcc -O2 -march=native benchmark.c generated_c/baseline_model.c generated_c/specialized_*.c -lm -o bench
 * Run:  ./bench test_rows.bin 2000000
 */
#ifndef __APPLE__
#define _POSIX_C_SOURCE 199309L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#ifndef GUARDS_FILE
#define GUARDS_FILE "generated_c/guards.c"
#endif
#include GUARDS_FILE   /* guard_calm / guard_normal / guard_volatile (static inline) */

double predict_baseline(double, double, double);
double predict_calm(double, double, double);
double predict_normal(double, double, double);
double predict_volatile(double, double, double);
#ifdef HAVE_FLAT
double predict_flat(double, double, double);
#endif

#define NF 3
#ifdef COLD_KB   /* evict caches (incl. code in L2) before every timed call by streaming a big buffer */
static unsigned char *cold_buf; static size_t cold_n; static volatile unsigned cold_sink;
static inline void cold_flush(void) { unsigned s = 0; for (size_t i = 0; i < cold_n; i += 128) s += cold_buf[i]; cold_sink += s; }
#define COLD_FLUSH() cold_flush()
#else
#define COLD_FLUSH() ((void)0)
#endif
#define EQ_TOL 1e-12

#ifdef __APPLE__
#include <mach/mach_time.h>
static inline uint64_t now_ns(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}
#else
static inline uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
#endif
static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}
static void report(const char *name, uint32_t *lat, size_t n, uint32_t ovh) {
    for (size_t i = 0; i < n; i++) lat[i] = lat[i] > ovh ? lat[i] - ovh : 0;
    qsort(lat, n, sizeof(uint32_t), cmp_u32);
    double sum = 0; for (size_t i = 0; i < n; i++) sum += lat[i];
    printf("%-10s mean %7.1f  p50 %5u  p99 %5u  p99.9 %5u  max %7u   (ns)\n",
           name, sum / n, lat[n / 2], lat[(size_t)(n * 0.99)], lat[(size_t)(n * 0.999)], lat[n - 1]);
}
/* 0..2 = regime whose guard passes, -1 = none (fallback to baseline) */
#ifndef SPEC_MASK
#define SPEC_MASK 7   /* bit0 calm, bit1 normal, bit2 volatile: which regimes are specialized; others fall back */
#endif
static inline int route(double a, double b, double c) {
    if ((SPEC_MASK & 1) && guard_calm(a, b, c)) return 0;
    if ((SPEC_MASK & 2) && guard_normal(a, b, c)) return 1;
    if ((SPEC_MASK & 4) && guard_volatile(a, b, c)) return 2;
    return -1;
}
static inline double call_spec(int r, double a, double b, double c) {
    return r == 0 ? predict_calm(a, b, c) : r == 1 ? predict_normal(a, b, c) : predict_volatile(a, b, c);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "test_rows.bin";
    size_t maxn = argc > 2 ? (size_t)atoll(argv[2]) : 2000000;
    FILE *f = fopen(path, "rb"); if (!f) { perror("open rows"); return 1; }
    float *Xf = malloc(maxn * NF * sizeof(float));
    size_t n = fread(Xf, NF * sizeof(float), maxn, f); fclose(f);
    double *X = malloc(n * NF * sizeof(double));
    for (size_t i = 0; i < n * NF; i++) X[i] = (double)Xf[i];
    free(Xf);
    printf("rows loaded: %zu\n", n);
#ifdef COLD_KB
    cold_n = (size_t)COLD_KB * 1024; cold_buf = malloc(cold_n);
    for (size_t i = 0; i < cold_n; i++) cold_buf[i] = (unsigned char)i;
    printf("COLD mode: %d KB streamed before every timed call\n", (int)COLD_KB);
#endif
    printf("regime mask: %d (1=calm 2=normal 4=volatile)\n", (int)SPEC_MASK);

    /* ---- 1. Guard-pass rate + equivalence ---- */
    const char *rn[3] = {"calm", "normal", "volatile"};
    size_t pass[3] = {0}, mism[3] = {0}; double maxdiff[3] = {0}; size_t none = 0;
    for (size_t i = 0; i < n; i++) {
        const double *x = X + i * NF; int r = route(x[0], x[1], x[2]);
        if (r < 0) { none++; continue; }
        pass[r]++;
        double d = fabs(call_spec(r, x[0], x[1], x[2]) - predict_baseline(x[0], x[1], x[2]));
        if (d > EQ_TOL) { mism[r]++; if (d > maxdiff[r]) maxdiff[r] = d; }
    }
    printf("\n== Guard-pass rate & equivalence (specialized vs baseline, guard-passing rows only) ==\n");
    size_t tp = 0, tm = 0;
    for (int r = 0; r < 3; r++) {
        printf("%-9s guard-pass %9zu (%5.2f%%)  mismatches %zu (%.4f%% of passes)  max|diff| %.3g\n",
               rn[r], pass[r], 100.0 * pass[r] / n, mism[r],
               pass[r] ? 100.0 * mism[r] / pass[r] : 0.0, maxdiff[r]);
        tp += pass[r]; tm += mism[r];
    }
    printf("fallback (no guard passed) %zu (%.2f%%)\n", none, 100.0 * none / n);
#ifdef HAVE_FLAT
    { size_t fm = 0; for (size_t i = 0; i < n; i++) { const double *x = X + i * NF;
        if (predict_flat(x[0], x[1], x[2]) != predict_baseline(x[0], x[1], x[2])) fm++; }
      printf("flat evaluator vs baseline: %zu mismatches / %zu rows\n", fm, n); }
#endif
    printf("EQUIVALENCE: %s (%zu mismatches / %zu guarded calls)\n", tm == 0 ? "EXACT" : "NOT EXACT", tm, tp);

    /* ---- 2. Latency (overall + split by path, same rows) ---- */
    uint32_t *lb = malloc(n * sizeof(uint32_t));   /* baseline latency per row  */
    uint32_t *ld = malloc(n * sizeof(uint32_t));   /* dispatch latency per row  */
    uint32_t *lg = malloc(n * sizeof(uint32_t));   /* guards-only latency       */
    uint32_t *tmp = malloc(n * sizeof(uint32_t));
    signed char *rt = malloc(n);
#ifdef HAVE_FLAT
    uint32_t *lf = malloc(n * sizeof(uint32_t)), *ldf = malloc(n * sizeof(uint32_t));
#endif
    volatile double sink = 0;
    for (size_t i = 0; i < n; i++) { uint64_t a = now_ns(); uint64_t b = now_ns(); tmp[i] = (uint32_t)(b - a); }
    qsort(tmp, n, sizeof(uint32_t), cmp_u32);
    uint32_t ovh = tmp[n / 2];
    printf("\n== Latency per call (chronological order, timer overhead %u ns subtracted) ==\n", ovh);

    for (size_t i = 0; i < n; i++) { const double *x = X + i * NF; sink += predict_baseline(x[0], x[1], x[2]); } /* warmup */

    for (size_t i = 0; i < n; i++) {
        const double *x = X + i * NF; COLD_FLUSH(); uint64_t a = now_ns();
        sink += predict_baseline(x[0], x[1], x[2]);
        uint64_t b = now_ns(); lb[i] = (uint32_t)(b - a);
    }
    for (size_t i = 0; i < n; i++) {
        const double *x = X + i * NF; COLD_FLUSH(); uint64_t a = now_ns();
        int r = route(x[0], x[1], x[2]);
        sink += (r < 0) ? predict_baseline(x[0], x[1], x[2]) : call_spec(r, x[0], x[1], x[2]);
        uint64_t b = now_ns(); ld[i] = (uint32_t)(b - a); rt[i] = (signed char)r;
    }
    for (size_t i = 0; i < n; i++) {
        const double *x = X + i * NF; COLD_FLUSH(); uint64_t a = now_ns();
        sink += route(x[0], x[1], x[2]);
        uint64_t b = now_ns(); lg[i] = (uint32_t)(b - a);
    }
#ifdef HAVE_FLAT
    for (size_t i = 0; i < n; i++) { const double *x = X + i * NF; sink += predict_flat(x[0], x[1], x[2]); } /* warmup */
    for (size_t i = 0; i < n; i++) {
        const double *x = X + i * NF; COLD_FLUSH(); uint64_t a = now_ns();
        sink += predict_flat(x[0], x[1], x[2]);
        uint64_t b = now_ns(); lf[i] = (uint32_t)(b - a);
    }
    for (size_t i = 0; i < n; i++) {   /* dispatch whose FALLBACK is the flat evaluator */
        const double *x = X + i * NF; COLD_FLUSH(); uint64_t a = now_ns();
        int r = route(x[0], x[1], x[2]);
        sink += (r < 0) ? predict_flat(x[0], x[1], x[2]) : call_spec(r, x[0], x[1], x[2]);
        uint64_t b = now_ns(); ldf[i] = (uint32_t)(b - a);
    }
#endif
    /* Cache-interference test: same dispatch, but rows ordered by path so each kernel's code stays warm */
    uint32_t *lgr = malloc(n * sizeof(uint32_t));
    size_t *ord = malloc(n * sizeof(size_t));
    { size_t m = 0;
      for (int g = -1; g <= 2; g++) for (size_t i = 0; i < n; i++) if (rt[i] == g) ord[m++] = i;
      for (size_t k = 0; k < n; k++) {
          size_t i = ord[k]; const double *x = X + i * NF; COLD_FLUSH(); uint64_t a = now_ns();
          int r = route(x[0], x[1], x[2]);
          sink += (r < 0) ? predict_baseline(x[0], x[1], x[2]) : call_spec(r, x[0], x[1], x[2]);
          uint64_t b = now_ns(); lgr[i] = (uint32_t)(b - a);
      } }
    #define SUBSET(name, arr, cond) do { size_t m = 0; \
        for (size_t i = 0; i < n; i++) if (cond) tmp[m++] = arr[i]; \
        if (m) report(name, tmp, m, ovh); } while (0)
    SUBSET("baseline", lb, 1);
    SUBSET("vec-disp", ld, 1);
    SUBSET("guards", lg, 1);
    printf("\n== Cache-interference test: same dispatch code, rows interleaved vs grouped by path ==\n");
    printf("-- fallback rows --\n");
    SUBSET("base-only", lb, rt[i] < 0);
    SUBSET("interleaved", ld, rt[i] < 0);
    SUBSET("grouped", lgr, rt[i] < 0);
    printf("-- guard-passed rows --\n");
    SUBSET("base-only", lb, rt[i] >= 0);
    SUBSET("interleaved", ld, rt[i] >= 0);
    SUBSET("grouped", lgr, rt[i] >= 0);
#ifdef HAVE_FLAT
    printf("\n== Strong-baseline comparison (flat branchless evaluator; vec+flatFB = VEC dispatch with flat fallback) ==\n");
    SUBSET("flat", lf, 1);
    SUBSET("vec-disp", ld, 1);
    SUBSET("vec+flatFB", ldf, 1);
    printf("-- guard-passed rows --\n");
    SUBSET("flat", lf, rt[i] >= 0);
    SUBSET("vec-disp", ld, rt[i] >= 0);
    printf("-- fallback rows --\n");
    SUBSET("flat", lf, rt[i] < 0);
    SUBSET("vec-disp", ld, rt[i] < 0);
    SUBSET("vec+flatFB", ldf, rt[i] < 0);
#endif

    printf("\n-- guard-PASSED rows only (%zu rows): the rows the specialization targets --\n", tp);
    SUBSET("baseline", lb, rt[i] >= 0);
    SUBSET("vec-disp", ld, rt[i] >= 0);
    for (int r = 0; r < 3; r++) {
        char nm[32], nd[32]; snprintf(nm, sizeof nm, "base/%s", rn[r]); snprintf(nd, sizeof nd, "disp/%s", rn[r]);
        SUBSET(nm, lb, rt[i] == r);
        SUBSET(nd, ld, rt[i] == r);
    }
    printf("\n-- FALLBACK rows only (%zu rows): pay guard cost, run baseline --\n", none);
    SUBSET("baseline", lb, rt[i] < 0);
    SUBSET("vec-disp", ld, rt[i] < 0);

    printf("\n-- per-regime guard-pass rows for the slowest 1%% of BASELINE calls --\n");
    {   /* where do the baseline's slowest calls come from? */
        for (size_t i = 0; i < n; i++) tmp[i] = lb[i];
        qsort(tmp, n, sizeof(uint32_t), cmp_u32);
        uint32_t thr = tmp[(size_t)(n * 0.99)]; size_t c[4] = {0}, tot = 0;
        for (size_t i = 0; i < n; i++) if (lb[i] > thr) { c[rt[i] + 1]++; tot++; }
        printf("calls above p99 (%u ns): %zu  | fallback %.1f%%  calm %.1f%%  normal %.1f%%  volatile %.1f%%\n",
               thr, tot, 100.0*c[0]/tot, 100.0*c[1]/tot, 100.0*c[2]/tot, 100.0*c[3]/tot);
    }
    (void)sink; free(X); free(lb); free(ld); free(lg); free(tmp); free(rt); free(lgr); free(ord);
    return 0;
}
