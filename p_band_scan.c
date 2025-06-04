#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>
#include <pthread.h>
#include <sched.h>

#include "filter.h"
#include "signal.h"
#include "timing.h"

#define MAXWIDTH 40
#define THRESHOLD 2.0
#define ALIENS_LOW  50000.0
#define ALIENS_HIGH 150000.0

typedef struct {
  int thread_id;
  int num_threads;
  int num_bands;
  int filter_order;
  double Fs;
  int num_samples;
  double* data;
  double* band_power;
  int num_processors;
} thread_arg_t;

void* band_worker(void* arg) {
  thread_arg_t* t = (thread_arg_t*) arg;

  // Set processor affinity
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(t->thread_id % t->num_processors, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

  double Fc = t->Fs / 2;
  double bandwidth = Fc / t->num_bands;

  double filter_coeffs[t->filter_order + 1];

  for (int band = t->thread_id; band < t->num_bands; band += t->num_threads) {
    generate_band_pass(t->Fs,
                       band * bandwidth + 0.0001,
                       (band + 1) * bandwidth - 0.0001,
                       t->filter_order,
                       filter_coeffs);

    hamming_window(t->filter_order, filter_coeffs);

    convolve_and_compute_power(t->num_samples,
                               t->data,
                               t->filter_order,
                               filter_coeffs,
                               &t->band_power[band]);
  }

  return NULL;
}

void remove_dc(double* data, int num) {
  double sum = 0;
  for (int i = 0; i < num; i++) sum += data[i];
  double avg = sum / num;
  for (int i = 0; i < num; i++) data[i] -= avg;
}

int main(int argc, char* argv[]) {
  if (argc != 8) {
    printf("usage: p_band_scan text|bin|mmap signal_file Fs filter_order num_bands num_threads num_processors\n");
    return -1;
  }

  char sig_type    = toupper(argv[1][0]);
  char* sig_file   = argv[2];
  double Fs        = atof(argv[3]);
  int filter_order = atoi(argv[4]);
  int num_bands    = atoi(argv[5]);
  int num_threads  = atoi(argv[6]);
  int num_procs    = atoi(argv[7]);

  assert(Fs > 0.0);
  assert(filter_order > 0 && !(filter_order & 0x1));
  assert(num_bands > 0);
  assert(num_threads > 0);
  assert(num_procs > 0);

  printf("type:     %s\nfile:     %s\nFs:       %lf Hz\norder:    %d\nbands:    %d\nthreads:  %d\nprocs:    %d\n",
         sig_type == 'T' ? "Text" : (sig_type == 'B' ? "Binary" : (sig_type == 'M' ? "Mapped Binary" : "UNKNOWN TYPE")),
         sig_file, Fs, filter_order, num_bands, num_threads, num_procs);

  signal* sig;
  switch (sig_type) {
    case 'T': sig = load_text_format_signal(sig_file); break;
    case 'B': sig = load_binary_format_signal(sig_file); break;
    case 'M': sig = map_binary_format_signal(sig_file); break;
    default:  printf("Unknown signal type\n"); return -1;
  }

  if (!sig) {
    printf("Unable to load or map file\n");
    return -1;
  }

  sig->Fs = Fs;
  remove_dc(sig->data, sig->num_samples);

  double band_power[num_bands];
  pthread_t threads[num_threads];
  thread_arg_t args[num_threads];

  for (int i = 0; i < num_threads; i++) {
    args[i] = (thread_arg_t){
      .thread_id = i,
      .num_threads = num_threads,
      .num_bands = num_bands,
      .filter_order = filter_order,
      .Fs = Fs,
      .num_samples = sig->num_samples,
      .data = sig->data,
      .band_power = band_power,
      .num_processors = num_procs
    };
    pthread_create(&threads[i], NULL, band_worker, &args[i]);
  }

  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }

  // Finish analysis like band_scan
  double Fc = Fs / 2;
  double bandwidth = Fc / num_bands;
  double max_band_power = band_power[0], sum = band_power[0];
  for (int i = 1; i < num_bands; i++) {
    if (band_power[i] > max_band_power) max_band_power = band_power[i];
    sum += band_power[i];
  }
  double avg_band_power = sum / num_bands;

  double lb = -1, ub = -1;
  int wow = 0;

  for (int band = 0; band < num_bands; band++) {
    double band_low = band * bandwidth + 0.0001;
    double band_high = (band + 1) * bandwidth - 0.0001;
    printf("%5d %20lf to %20lf Hz: %20lf ", band, band_low, band_high, band_power[band]);
    for (int i = 0; i < MAXWIDTH * (band_power[band] / max_band_power); i++) printf("*");

    if ((band_low >= ALIENS_LOW && band_low <= ALIENS_HIGH) ||
        (band_high >= ALIENS_LOW && band_high <= ALIENS_HIGH)) {
      if (band_power[band] > THRESHOLD * avg_band_power) {
        printf("(WOW)");
        wow = 1;
        if (lb < 0) lb = band_low;
        ub = band_high;
      } else {
        printf("(meh)");
      }
    } else {
      printf("(meh)");
    }
    printf("\n");
  }

  if (wow) {
    printf("POSSIBLE ALIENS %lf-%lf HZ (CENTER %lf HZ)\n", lb, ub, (ub + lb) / 2.0);
  } else {
    printf("no aliens\n");
  }

  free_signal(sig);
  return 0;
}
