#-*- makefile -*-
# You can pick a different compiler here
# and also choose different options

CC = gcc -g -Wall
AR = ar

all: libfilter.a band_scan pthread-ex parallel-sum-ex p_band_scan

libfilter.a : filter.o signal.o timing.o
	$(AR) ruv libfilter.a filter.o signal.o timing.o

filter.o : filter.c filter.h
	$(CC) -c filter.c

signal.o : signal.c signal.h
	$(CC) -c signal.c

timing.o : timing.c timing.h
	$(CC) -c timing.c


band_scan: band_scan.c filter.h signal.h timing.h libfilter.a
	$(CC) band_scan.c -L. -lfilter -lm -o band_scan

p_band_scan: p_band_scan.c
	gcc -O3 -pthread -o p_band_scan p_band_scan.c signal.c filter.c timing.c -lm
#
# Your rule for p_band_scan will look like the following.  Note the use of the
# -pthread option which is critical
#
# You could add p_band_scan to the "all:" rule above so it runs by default
#
#
#p_band_scan: p_band_scan.c filter.h signal.h timing.h libfilter.a
#	    $(CC) -pthread p_band_scan.c -L. -lfilter -lm -o p_band_scan
#

clean-filter:
	-rm filter.o signal.o timing.o libfilter.a  band_scan 2>/dev/null || true

.PHONY: clean-filter

parallel-sum-ex: parallel-sum-ex.c
	$(CC) -pthread parallel-sum-ex.c -o parallel-sum-ex

pthread-ex: pthread-ex.c
	$(CC) -pthread pthread-ex.c -o pthread-ex

clean-examples:
	-rm -f pthread-ex parallel-sum-ex 2>/dev/null || true

.PHONY: clean-examples


clean: clean-filter clean-examples

