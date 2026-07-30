/*
 * cagent — minimal Claude Code-like agent in C (~400 lines)
 *
 * Talks to an OpenAI-compatible /v1/chat/completions endpoint
 * (e.g. mlx-lm server running Qwen3.6 / Nemotron) using native
 * OpenAI tool calling, and exposes 4 local tools:
 * bash / read_file / write_file / edit_file.
 *
 * NOTE: mlx-lm >= 0.31 parses the model's <tool_call> output server-side;
 * a prompt-only text protocol gets swallowed (empty message). Always pass
 * "tools" in the request and read message.tool_calls.
 *
 * usage:
 *   agent [-p "prompt"] [-y] [-t] [-m model] [-b base_url]
 *     -p  one-shot mode: run a single task and exit (final answer on stdout)
 *     -y  auto-approve tool executions (no [y/N] prompt)
 *     -t  thinking mode: let the model reason before acting
 *         (slower, ~5-10x tokens, noticeably better tool choices)
 *
 * env (flags take precedence):
 *   AGENT_BASE   default http://127.0.0.1:8780 (any OpenAI-compatible server;
 *                for a remote host: ssh -L 8780:127.0.0.1:8780 user@llm-host)
 *                "teai" = https://api.teai.io (hosted, 100+ models, anon OK)
 *   AGENT_MODEL  default mlx-community/Qwen3.6-35B-A3B-4bit
 *                (teai.io base: defaults to "teai/auto" auto-router instead)
 *   AGENT_KEY    optional Bearer token (teai.io: te_... API key for the
 *                full catalog + credits; anonymous = free tier)
 *   AGENT_THINK  set to keep <think> mode (default: enable_thinking=false)
 *
 * voice (KOE, koe.live):
 *   -k           voice mode: speak final answers aloud; in the REPL, type
 *                "v" + Enter to talk instead of typing (mic -> STT -> prompt)
 *   --koe-enroll [id]  record your voice once and register it on the spot
 *                (requires KOE_KEY; prints the voice id to use as KOE_VOICE)
 *   KOE_BASE     default https://koe.live
 *   KOE_KEY      koe.live admin/MCP token (X-Koe-Admin). Optional: without
 *                it, speaking with public consented voices + STT still work
 *   KOE_VOICE    voice id for replies (default "kentaro"; your own id after
 *                --koe-enroll)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <signal.h>
#include <fcntl.h>
#include <curl/curl.h>
#include "cJSON.h"

#define MAX_TOOL_OUT (64 * 1024)
#define MAX_FILE     (4 * 1024 * 1024)
#define MAX_HOPS     50
#define KEEP_TAIL    8     /* recent messages kept verbatim when trimming */
#define TRIM_OVER    600   /* tool results longer than this get trimmed   */

static int g_yes = 0;      /* -y: auto-approve tools */
static int g_koe = 0;      /* -k: speak answers / mic input via koe.live */
static int g_think = 0;    /* -t: enable model reasoning before actions */
static char g_finish[32];  /* finish_reason of the last LLM call */

static const char *SYSTEM_PROMPT =
    "You are cagent, a minimal coding agent in a terminal. Use the provided "
    "tools (bash, read_file, write_file, edit_file) to accomplish the user's "
    "task. Verify your work with tools instead of assuming success. When the "
    "task is complete, give a final plain-text answer with no tool call; if "
    "the user asked for a specific value, the final answer must contain it. "
    "Reply in the user's language.";

static const char *TOOLS_JSON =
    "["
    "{\"type\":\"function\",\"function\":{\"name\":\"bash\","
    "\"description\":\"Run a shell command and return stdout+stderr.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},"
    "\"required\":[\"command\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"read_file\","
    "\"description\":\"Read a text file.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},"
    "\"required\":[\"path\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"write_file\","
    "\"description\":\"Create or overwrite a file.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
    "\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit_file\","
    "\"description\":\"Replace the first occurrence of old_string with new_string in a file.\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
    "\"old_string\":{\"type\":\"string\"},\"new_string\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"old_string\",\"new_string\"]}}}"
    "]";

/* ---------- helpers ---------- */

static const char *jstr(cJSON *o, const char *k) {
    cJSON *v = o ? cJSON_GetObjectItem(o, k) : NULL;
    return (v && cJSON_IsString(v)) ? v->valuestring : NULL;
}

/* shorten n so the buffer doesn't end mid-UTF-8-sequence — an invalid byte
 * sequence in the request body crashes mlx_lm.server's JSON decode */
static size_t utf8_trim(const char *s, size_t n) {
    size_t i = n;
    while (i > 0 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) i--;
    if (i > 0) {
        unsigned char lead = (unsigned char)s[i - 1];
        size_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
        if (need > n - (i - 1)) n = i - 1;
    }
    return n;
}

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + tv.tv_usec / 1e6;
}

/* ---------- HTTP ---------- */

typedef struct { char *data; size_t len; } Buf;

static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *ud) {
    Buf *b = (Buf *)ud;
    size_t n = size * nmemb;
    char *p = realloc(b->data, b->len + n + 1);
    if (!p) return 0;
    b->data = p;
    memcpy(b->data + b->len, ptr, n);
    b->len += n;
    b->data[b->len] = 0;
    return n;
}

static char *http_post(const char *url, const char *body) {
    for (int attempt = 0; attempt < 4; attempt++) {
        CURL *curl = curl_easy_init();
        if (!curl) return NULL;
        Buf resp = {0};
        struct curl_slist *hdrs = curl_slist_append(NULL, "Content-Type: application/json");
        const char *key = getenv("AGENT_KEY");
        if (key && *key) {
            char auth[512];
            snprintf(auth, sizeof auth, "Authorization: Bearer %s", key);
            hdrs = curl_slist_append(hdrs, auth);
        }
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L);
        CURLcode rc = curl_easy_perform(curl);
        curl_slist_free_all(hdrs);
        curl_easy_cleanup(curl);
        if (rc == CURLE_OK) return resp.data;
        fprintf(stderr, "http: %s%s\n", curl_easy_strerror(rc),
                attempt < 3 ? " (retrying)" : "");
        free(resp.data);
        if (attempt < 3) sleep(2u << attempt);
    }
    return NULL;
}

/* ---------- LLM call ---------- */

/* tool results older than the last KEEP_TAIL messages rarely matter verbatim;
 * shrink them so long sessions don't blow up the context window */
static void trim_history(cJSON *messages) {
    int n = cJSON_GetArraySize(messages);
    for (int i = 1; i < n - KEEP_TAIL; i++) {
        cJSON *m = cJSON_GetArrayItem(messages, i);
        const char *role = jstr(m, "role");
        cJSON *content = cJSON_GetObjectItem(m, "content");
        if (role && !strcmp(role, "tool") && content && cJSON_IsString(content)
            && strlen(content->valuestring) > TRIM_OVER) {
            char keep[400];
            size_t cut = utf8_trim(content->valuestring, 300);
            snprintf(keep, sizeof keep, "%.*s\n...[trimmed]",
                     (int)cut, content->valuestring);
            cJSON_SetValuestring(content, keep);
        }
    }
}

/* returns a detached copy of choices[0].message, or NULL */
static cJSON *api_call(cJSON *messages, cJSON *tools, const char *base, const char *model) {
    trim_history(messages);
    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "model", model);
    cJSON_AddItemToObject(req, "messages", cJSON_Duplicate(messages, 1));
    cJSON_AddItemToObject(req, "tools", cJSON_Duplicate(tools, 1));
    cJSON_AddNumberToObject(req, "max_tokens", g_think ? 16384 : 8192);
    cJSON_AddNumberToObject(req, "temperature", 0.2);
    /* Qwen3.x on mlx defaults to <think> mode; control it explicitly */
    cJSON *kw = cJSON_AddObjectToObject(req, "chat_template_kwargs");
    cJSON_AddBoolToObject(kw, "enable_thinking", g_think);
    char *body = cJSON_PrintUnformatted(req);
    cJSON_Delete(req);

    char url[512];
    snprintf(url, sizeof url, "%s/v1/chat/completions", base);
    double t0 = now_s();
    char *raw = http_post(url, body);
    double dt = now_s() - t0;
    free(body);
    if (!raw) return NULL;

    cJSON *out = NULL;
    cJSON *resp = cJSON_Parse(raw);
    if (resp) {
        /* OpenAI-style error envelope: surface it plainly instead of the raw
         * JSON dump, and point at how to unblock (a key = higher limits). */
        cJSON *err = cJSON_GetObjectItem(resp, "error");
        if (err && !cJSON_GetObjectItem(resp, "choices")) {
            const char *emsg = cJSON_IsString(err) ? err->valuestring : jstr(err, "message");
            const char *ecode = cJSON_IsObject(err) ? jstr(err, "code") : NULL;
            fprintf(stderr, "api error: %s\n", emsg ? emsg : "unknown");
            int limit_hit = (ecode && (strstr(ecode, "rate_limit") || strstr(ecode, "daily_limit")))
                || (emsg && (strstr(emsg, "limit") || strstr(emsg, "上限")));
            if (limit_hit && !getenv("AGENT_KEY") && strstr(base, "teai.io")) {
                fprintf(stderr,
                    "  → 無料枠の上限です。teai.io でAPIキーを取ると解除されます:\n"
                    "      1. https://teai.io で登録 (メール+パスワード)\n"
                    "      2. 設定画面でAPIキー(te_...)を発行\n"
                    "      3. export AGENT_KEY=te_...   # 100+モデル・クレジット課金\n");
            }
            cJSON_Delete(resp);
            free(raw);
            return NULL;
        }
        cJSON *choices = cJSON_GetObjectItem(resp, "choices");
        cJSON *c0      = choices ? cJSON_GetArrayItem(choices, 0) : NULL;
        cJSON *msg     = c0 ? cJSON_GetObjectItem(c0, "message") : NULL;
        if (msg) out = cJSON_Duplicate(msg, 1);
        const char *fr = jstr(c0, "finish_reason");
        snprintf(g_finish, sizeof g_finish, "%s", fr ? fr : "");
        cJSON *usage = cJSON_GetObjectItem(resp, "usage");
        if (usage) {
            cJSON *pt = cJSON_GetObjectItem(usage, "prompt_tokens");
            cJSON *ct = cJSON_GetObjectItem(usage, "completion_tokens");
            fprintf(stderr, "  [llm %.1fs in=%d out=%d]\n", dt,
                    pt ? (int)pt->valuedouble : -1, ct ? (int)ct->valuedouble : -1);
        }
    }
    if (!out) fprintf(stderr, "api: unexpected response: %.400s\n", raw);
    cJSON_Delete(resp);
    free(raw);
    return out;
}

static void strip_think(char *s) {
    /* Reasoning models (Nemotron via teai.io, Qwen local) sometimes put the
     * whole answer INSIDE <think> and leave nothing after it. If stripping
     * would erase everything, salvage the think body instead of going mute. */
    char *first = strstr(s, "<think>");
    if (first) {
        char probe_ok = 0;
        for (char *p = s; *p; ) {
            char *a = strstr(p, "<think>");
            if (!a) { if (*p) probe_ok = 1; break; }
            if (a != p) { probe_ok = 1; break; }   /* text before the block */
            char *b = strstr(a, "</think>");
            if (!b) break;
            p = b + strlen("</think>");
            while (*p == ' ' || *p == '\n') p++;
        }
        if (!probe_ok) {
            /* keep inner text of the first think block as the visible answer */
            char *inner = first + strlen("<think>");
            char *end = strstr(inner, "</think>");
            size_t len = end ? (size_t)(end - inner) : strlen(inner);
            memmove(s, inner, len);
            s[len] = 0;
            return;
        }
    }
    for (;;) {
        char *a = strstr(s, "<think>");
        if (!a) return;
        char *b = strstr(a, "</think>");
        if (!b) { *a = 0; return; }
        b += strlen("</think>");
        memmove(a, b, strlen(b) + 1);
    }
}

/* ---------- KOE voice I/O (koe.live) ---------- */

static const char *koe_base(void) {
    const char *b = getenv("KOE_BASE");
    return (b && *b) ? b : "https://koe.live";
}

/* binary-safe POST; ctype = request Content-Type; KOE_KEY -> X-Koe-Admin */
static Buf koe_post(const char *path, const void *body, size_t body_len, const char *ctype) {
    Buf resp = {0};
    CURL *curl = curl_easy_init();
    if (!curl) return resp;
    char url[512], ct[128];
    snprintf(url, sizeof url, "%s%s", koe_base(), path);
    snprintf(ct, sizeof ct, "Content-Type: %s", ctype);
    struct curl_slist *hdrs = curl_slist_append(NULL, ct);
    const char *key = getenv("KOE_KEY");
    if (key && *key) {
        char auth[512];
        snprintf(auth, sizeof auth, "X-Koe-Admin: %s", key);
        hdrs = curl_slist_append(hdrs, auth);
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body_len);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK) { free(resp.data); resp.data = NULL; resp.len = 0; }
    return resp;
}

static const char *find_cmd(const char *const *cands) {
    static char found[64];
    for (int i = 0; cands[i]; i++) {
        char probe[128];
        snprintf(probe, sizeof probe, "command -v %s >/dev/null 2>&1", cands[i]);
        if (system(probe) == 0) { snprintf(found, sizeof found, "%s", cands[i]); return found; }
    }
    return NULL;
}

/* strip code fences / URLs and cap length so TTS reads prose, not diffs */
static void koe_sanitize(const char *in, char *out, size_t cap) {
    size_t o = 0;
    int fence = 0;
    for (const char *p = in; *p && o + 8 < cap; ) {
        if (!strncmp(p, "```", 3)) { fence = !fence; p += 3; continue; }
        if (fence) { p++; continue; }
        if (!strncmp(p, "http://", 7) || !strncmp(p, "https://", 8)) {
            while (*p && *p != ' ' && *p != '\n' && *p != ')') p++;
            o += snprintf(out + o, cap - o, "リンク");
            continue;
        }
        if (*p == '`' || *p == '*' || *p == '#') { p++; continue; }
        out[o++] = *p++;
    }
    out[utf8_trim(out, o)] = 0;
}

/* speak text aloud via koe.live (blocking; failures are non-fatal) */
static void koe_say(const char *text) {
    if (!g_koe || !text || !*text) return;
    char clean[720];
    koe_sanitize(text, clean, sizeof clean);
    if (!clean[0]) return;
    const char *voice = getenv("KOE_VOICE");
    if (!voice || !*voice) voice = "kentaro";
    cJSON *req = cJSON_CreateObject();
    cJSON_AddStringToObject(req, "text", clean);
    cJSON_AddStringToObject(req, "user_id", voice);
    cJSON_AddStringToObject(req, "source", "cagent");
    char *body = cJSON_PrintUnformatted(req);
    cJSON_Delete(req);
    Buf r = koe_post("/api/speak", body, strlen(body), "application/json");
    free(body);
    if (!r.data) { fprintf(stderr, "  [koe: no response]\n"); return; }
    if (r.len && r.data[0] == '{') {              /* JSON error, not audio */
        cJSON *e = cJSON_Parse(r.data);
        fprintf(stderr, "  [koe: %s]\n", jstr(e, "detail") ? jstr(e, "detail") : "error");
        cJSON_Delete(e); free(r.data); return;
    }
    char mp3[128];
    snprintf(mp3, sizeof mp3, "/tmp/cagent_say_%d.mp3", getpid());
    FILE *f = fopen(mp3, "wb");
    if (f) { fwrite(r.data, 1, r.len, f); fclose(f); }
    free(r.data);
    static const char *players[] = { "afplay", "mpg123", "ffplay", NULL };
    const char *pl = find_cmd(players);
    if (!pl) { fprintf(stderr, "  [koe: no audio player (afplay/mpg123/ffplay)]\n"); return; }
    char cmd[256];
    if (!strcmp(pl, "ffplay"))
        snprintf(cmd, sizeof cmd, "ffplay -nodisp -autoexit -loglevel quiet %s", mp3);
    else if (!strcmp(pl, "mpg123"))
        snprintf(cmd, sizeof cmd, "mpg123 -q %s", mp3);
    else
        snprintf(cmd, sizeof cmd, "afplay %s", mp3);
    system(cmd);
    unlink(mp3);
}

/* record from the mic until Enter (wav 16k mono). returns 0 on success */
static int koe_record(const char *wav) {
    static const char *recs[] = { "ffmpeg", "rec", "arecord", NULL };
    const char *rc = find_cmd(recs);
    if (!rc) { fprintf(stderr, "  [koe: no recorder — brew install ffmpeg or sox]\n"); return -1; }
    fprintf(stderr, "  🎤 録音中… Enterで停止\n");
    pid_t pid = fork();
    if (pid == 0) {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, 2); }
        if (!strcmp(rc, "ffmpeg"))
            execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-f", "avfoundation", "-i", ":0", "-ac", "1", "-ar", "16000",
                   "-y", wav, (char *)NULL);
        else if (!strcmp(rc, "rec"))
            execlp("rec", "rec", "-q", "-c", "1", "-r", "16000", wav, (char *)NULL);
        else
            execlp("arecord", "arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", wav, (char *)NULL);
        _exit(127);
    }
    char tmp[16];
    if (!fgets(tmp, sizeof tmp, stdin)) { /* EOF: stop anyway */ }
    kill(pid, SIGINT);
    int st; waitpid(pid, &st, 0);
    FILE *f = fopen(wav, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n > 4000 ? 0 : -1;   /* <0.25s of audio = treat as failed */
}

/* mic -> koe.live STT -> heard text (malloc'd, or NULL) */
static char *koe_listen(void) {
    char wav[128];
    snprintf(wav, sizeof wav, "/tmp/cagent_rec_%d.wav", getpid());
    if (koe_record(wav) != 0) { unlink(wav); return NULL; }
    FILE *f = fopen(wav, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *audio = malloc(n);
    if (!audio || fread(audio, 1, n, f) != (size_t)n) { fclose(f); free(audio); unlink(wav); return NULL; }
    fclose(f); unlink(wav);
    Buf r = koe_post("/api/stt", audio, n, "audio/wav");
    free(audio);
    if (!r.data) return NULL;
    cJSON *j = cJSON_Parse(r.data);
    char *heard = jstr(j, "text") ? strdup(jstr(j, "text")) : NULL;
    if (!heard) fprintf(stderr, "  [koe stt: %s]\n", jstr(j, "detail") ? jstr(j, "detail") : "failed");
    cJSON_Delete(j); free(r.data);
    return heard;
}

/* record once -> register your own voice on the spot (needs KOE_KEY) */
static int koe_enroll(const char *want_id) {
    if (!getenv("KOE_KEY") || !*getenv("KOE_KEY")) {
        fprintf(stderr, "koe-enroll: KOE_KEY が必要です(声の登録=同意ゲート付きのアカウント機能)。\n"
                        "koe.live でログイン/キー発行してから再実行してください。\n");
        return 1;
    }
    char body[256];
    snprintf(body, sizeof body, "{\"user_id\":\"%s\"}", want_id && *want_id ? want_id : "cagent");
    Buf r = koe_post("/api/voice/register/start", body, strlen(body), "application/json");
    if (!r.data) { fprintf(stderr, "koe-enroll: start failed\n"); return 1; }
    cJSON *j = cJSON_Parse(r.data);
    const char *token = jstr(j, "token"), *phrase = jstr(j, "phrase");
    if (!token || !phrase) {
        fprintf(stderr, "koe-enroll: %s\n", jstr(j, "detail") ? jstr(j, "detail") : r.data);
        cJSON_Delete(j); free(r.data); return 1;
    }
    fprintf(stderr, "\n  次のお題を、そのまま声に出して読んでください:\n  「%s」\n\n", phrase);
    char tok[128]; snprintf(tok, sizeof tok, "%s", token);
    cJSON_Delete(j); free(r.data);

    char wav[128];
    snprintf(wav, sizeof wav, "/tmp/cagent_enroll_%d.wav", getpid());
    if (koe_record(wav) != 0) { fprintf(stderr, "koe-enroll: 録音に失敗しました\n"); return 1; }

    char b64cmd[256];
    snprintf(b64cmd, sizeof b64cmd, "base64 < %s | tr -d '\\n'", wav);
    FILE *bp = popen(b64cmd, "r");
    if (!bp) { unlink(wav); return 1; }
    size_t cap = 4 * 1024 * 1024, blen = 0;
    char *b64 = malloc(cap);
    blen = fread(b64, 1, cap - 1, bp);
    b64[blen] = 0;
    pclose(bp); unlink(wav);

    cJSON *vreq = cJSON_CreateObject();
    cJSON_AddStringToObject(vreq, "token", tok);
    cJSON_AddStringToObject(vreq, "audio_b64", b64);
    free(b64);
    char *vbody = cJSON_PrintUnformatted(vreq);
    cJSON_Delete(vreq);
    Buf vr = koe_post("/api/voice/register/verify", vbody, strlen(vbody), "application/json");
    free(vbody);
    if (!vr.data) { fprintf(stderr, "koe-enroll: verify failed\n"); return 1; }
    cJSON *vj = cJSON_Parse(vr.data);
    const char *uid = jstr(vj, "uid");
    if (uid) {
        printf("✅ 声を登録しました: %s\n次からはこの声で読み上げます:\n  export KOE_VOICE=%s\n", uid, uid);
    } else {
        fprintf(stderr, "koe-enroll: %s\n", jstr(vj, "detail") ? jstr(vj, "detail") : vr.data);
    }
    int ok = uid ? 0 : 1;
    cJSON_Delete(vj); free(vr.data);
    return ok;
}

/* ---------- tools ---------- */

static int confirm(void) {
    if (g_yes) { fprintf(stderr, "[auto-yes]\n"); return 1; }
    fprintf(stderr, "[y/N] > ");
    char line[32];
    if (!fgets(line, sizeof line, stdin)) return 0;
    return line[0] == 'y' || line[0] == 'Y';
}

static char *read_whole(const char *path, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = malloc(cap + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, cap, f);
    fclose(f);
    if (n == cap) n = utf8_trim(buf, n);
    buf[n] = 0;
    return buf;
}

static char *tool_bash(const char *cmd) {
    char *full = malloc(strlen(cmd) + 8);
    sprintf(full, "%s 2>&1", cmd);
    FILE *p = popen(full, "r");
    free(full);
    if (!p) return strdup("error: popen failed");
    char *out = malloc(MAX_TOOL_OUT + 64);
    size_t n = 0;
    while (n < MAX_TOOL_OUT) {
        size_t r = fread(out + n, 1, MAX_TOOL_OUT - n, p);
        if (r == 0) break;
        n += r;
    }
    int st = pclose(p);
    if (n == MAX_TOOL_OUT) n = utf8_trim(out, n);
    sprintf(out + n, "\n[exit %d]", WIFEXITED(st) ? WEXITSTATUS(st) : -1);
    return out;
}

static char *tool_read(const char *path) {
    char *s = read_whole(path, MAX_TOOL_OUT);
    return s ? s : strdup("error: cannot open file");
}

static char *tool_write(const char *path, const char *content) {
    FILE *f = fopen(path, "wb");
    if (!f) return strdup("error: cannot open for write");
    size_t len = strlen(content);
    fwrite(content, 1, len, f);
    fclose(f);
    char *r = malloc(strlen(path) + 64);
    sprintf(r, "wrote %zu bytes to %s", len, path);
    return r;
}

static char *tool_edit(const char *path, const char *olds, const char *news) {
    char *s = read_whole(path, MAX_FILE);
    if (!s) return strdup("error: cannot open file");
    char *pos = strstr(s, olds);
    if (!pos) { free(s); return strdup("error: old_string not found"); }
    FILE *f = fopen(path, "wb");
    if (!f) { free(s); return strdup("error: cannot open for write"); }
    fwrite(s, 1, (size_t)(pos - s), f);
    fwrite(news, 1, strlen(news), f);
    char *rest = pos + strlen(olds);
    fwrite(rest, 1, strlen(rest), f);
    fclose(f);
    free(s);
    return strdup("edit applied");
}

static char *run_tool(const char *name, cJSON *args) {
    if (!strcmp(name, "bash")) {
        const char *cmd = jstr(args, "command");
        if (!cmd) return strdup("error: missing command");
        fprintf(stderr, "\n  $ %s\n  ", cmd);
        if (!confirm()) return strdup("user denied permission");
        return tool_bash(cmd);
    }
    if (!strcmp(name, "read_file")) {
        const char *path = jstr(args, "path");
        if (!path) return strdup("error: missing path");
        fprintf(stderr, "\n  read %s\n", path);
        return tool_read(path);
    }
    if (!strcmp(name, "write_file")) {
        const char *path = jstr(args, "path");
        const char *content = jstr(args, "content");
        if (!path || !content) return strdup("error: missing path/content");
        fprintf(stderr, "\n  write %s (%zu bytes)\n  ", path, strlen(content));
        if (!confirm()) return strdup("user denied permission");
        return tool_write(path, content);
    }
    if (!strcmp(name, "edit_file")) {
        const char *path = jstr(args, "path");
        const char *olds = jstr(args, "old_string");
        const char *news = jstr(args, "new_string");
        if (!path || !olds || !news) return strdup("error: missing args");
        fprintf(stderr, "\n  edit %s\n  ", path);
        if (!confirm()) return strdup("user denied permission");
        return tool_edit(path, olds, news);
    }
    return strdup("error: unknown tool");
}

/* ---------- agent loop ---------- */

static void push_msg(cJSON *messages, const char *role, const char *content) {
    cJSON *m = cJSON_CreateObject();
    cJSON_AddStringToObject(m, "role", role);
    cJSON_AddStringToObject(m, "content", content);
    cJSON_AddItemToArray(messages, m);
}

static void agent_turn(cJSON *messages, cJSON *tools, const char *base, const char *model) {
    char last_call[512] = "";
    int repeat = 0;
    for (int hop = 0; hop < MAX_HOPS; hop++) {
        cJSON *msg = api_call(messages, tools, base, model);
        if (!msg) return;

        cJSON *content = cJSON_GetObjectItem(msg, "content");
        char *ctext = (content && cJSON_IsString(content)) ? content->valuestring : NULL;
        if (ctext) strip_think(ctext); /* shrinks in place, safe for cJSON */

        cJSON_AddItemToArray(messages, msg); /* echo assistant msg back verbatim */

        cJSON *tcs = cJSON_GetObjectItem(msg, "tool_calls");
        if (!tcs || cJSON_GetArraySize(tcs) == 0) {
            if (!strcmp(g_finish, "length")) {
                /* output hit max_tokens mid-generation (often a huge tool
                 * call); tell the model to continue in smaller steps */
                fprintf(stderr, "  [truncated at max_tokens — asking model to split the work]\n");
                push_msg(messages, "user",
                         "(system: your output was truncated at the token limit and no tool "
                         "call was executed. The task is NOT done. Continue with smaller "
                         "steps: write files in chunks of at most 120 lines, using "
                         "write_file for the first chunk and bash 'cat >> file' heredocs "
                         "or edit_file for the rest.)");
                continue;
            }
            printf("%s\n", ctext ? ctext : "(empty response)");
            if (ctext) koe_say(ctext);
            return;
        }
        if (ctext && ctext[0]) fprintf(stderr, "%s\n", ctext); /* narration */

        cJSON *tc;
        cJSON_ArrayForEach(tc, tcs) {
            cJSON *fn = cJSON_GetObjectItem(tc, "function");
            const char *name = jstr(fn, "name");
            const char *id   = jstr(tc, "id");
            cJSON *argsit = fn ? cJSON_GetObjectItem(fn, "arguments") : NULL;
            cJSON *args = NULL;
            if (argsit && cJSON_IsString(argsit)) args = cJSON_Parse(argsit->valuestring);
            else if (argsit)                      args = cJSON_Duplicate(argsit, 1);

            /* loop breaker: small local models tend to repeat the same
             * verification command forever instead of acting on it */
            char sig[512];
            char *argstr = args ? cJSON_PrintUnformatted(args) : strdup("");
            snprintf(sig, sizeof sig, "%s|%.400s", name ? name : "?", argstr);
            free(argstr);
            repeat = strcmp(sig, last_call) ? 0 : repeat + 1;
            snprintf(last_call, sizeof last_call, "%s", sig);

            char *result;
            if (repeat >= 2) {
                fprintf(stderr, "  [loop breaker: same call %dx]\n", repeat + 1);
                result = strdup("(system: you have now issued this exact tool call "
                    "3+ times in a row and the answer will not change. STOP verifying. "
                    "Either perform the actual edit/work the task asked for, or give "
                    "your final answer now.)");
            } else {
                result = run_tool(name ? name : "?", args);
            }
            cJSON_Delete(args);

            cJSON *tm = cJSON_CreateObject();
            cJSON_AddStringToObject(tm, "role", "tool");
            cJSON_AddStringToObject(tm, "tool_call_id", id ? id : "");
            cJSON_AddStringToObject(tm, "content", result);
            free(result);
            cJSON_AddItemToArray(messages, tm);
        }
    }
    fprintf(stderr, "agent: hop limit (%d) reached\n", MAX_HOPS);
}

int main(int argc, char **argv) {
    const char *base    = getenv("AGENT_BASE");
    const char *model   = getenv("AGENT_MODEL");
    const char *oneshot = NULL;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-y")) g_yes = 1;
        else if (!strcmp(argv[i], "-t")) g_think = 1;
        else if (!strcmp(argv[i], "-k")) g_koe = 1;
        else if (!strcmp(argv[i], "--koe-enroll")) {
            const char *id = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[++i] : "";
            curl_global_init(CURL_GLOBAL_DEFAULT);
            return koe_enroll(id);
        }
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) oneshot = argv[++i];
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) model   = argv[++i];
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) base    = argv[++i];
        else {
            fprintf(stderr, "usage: agent [-p \"prompt\"] [-y] [-t] [-k] [-m model] [-b base_url] [--koe-enroll [id]]\n");
            return 1;
        }
    }
    if (getenv("AGENT_THINK")) g_think = 1;
    if (!base)  base  = "http://127.0.0.1:8780";
    /* teai.io shorthand: AGENT_BASE=teai (or -b teai) → hosted multi-model API.
     * Anonymous works (free tier); AGENT_KEY=te_... unlocks the full catalog. */
    if (!strcmp(base, "teai")) base = "https://api.teai.io";
    if (!model) model = strstr(base, "teai.io")
        ? "teai/auto"                          /* server-side auto-router */
        : "mlx-community/Qwen3.6-35B-A3B-4bit";
    curl_global_init(CURL_GLOBAL_DEFAULT);

    char cwd[1024] = "?";
    getcwd(cwd, sizeof cwd);
    char sys[2048];
    snprintf(sys, sizeof sys, "%s\nCurrent working directory: %s", SYSTEM_PROMPT, cwd);

    cJSON *tools = cJSON_Parse(TOOLS_JSON);
    cJSON *messages = cJSON_CreateArray();
    push_msg(messages, "system", sys);

    if (oneshot) {
        push_msg(messages, "user", oneshot);
        agent_turn(messages, tools, base, model);
    } else {
        fprintf(stderr, "cagent — %s @ %s ('exit' to quit%s)\n", model, base,
                g_koe ? ", 'v' to talk" : "");
        char line[16384];
        for (;;) {
            fprintf(stderr, "\nyou> ");
            if (!fgets(line, sizeof line, stdin)) break;
            line[strcspn(line, "\n")] = 0;
            if (!line[0]) continue;
            if (!strcmp(line, "exit")) break;
            if (g_koe && !strcmp(line, "v")) {          /* voice turn */
                char *heard = koe_listen();
                if (!heard) continue;
                fprintf(stderr, "  🎤 「%s」\n", heard);
                push_msg(messages, "user", heard);
                free(heard);
                agent_turn(messages, tools, base, model);
                continue;
            }
            push_msg(messages, "user", line);
            agent_turn(messages, tools, base, model);
        }
    }
    cJSON_Delete(messages);
    cJSON_Delete(tools);
    curl_global_cleanup();
    return 0;
}
