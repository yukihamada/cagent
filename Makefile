CC      = cc
CFLAGS  = -O2 -Wall -Wextra
LDLIBS  = -lcurl

agent: agent.c cJSON.c cJSON.h
	$(CC) $(CFLAGS) -o agent agent.c cJSON.c $(LDLIBS)

clean:
	rm -f agent

.PHONY: clean
