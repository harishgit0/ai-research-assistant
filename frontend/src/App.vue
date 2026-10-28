<script setup>
import { computed, onMounted, ref } from "vue";

const API = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const documents = ref([]);
const scope = ref("");
const file = ref(null);
const question = ref("");
const result = ref(null);
const error = ref("");
const notice = ref("");
const busy = ref("");
const canAsk = computed(() => question.value.trim() && !busy.value && documents.value.length);

async function api(path, options = {}) {
  const response = await fetch(API + path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "Request failed (" + response.status + ").");
  return body;
}
async function refresh() {
  try {
    documents.value = await api("/documents");
    if (scope.value && !documents.value.some((d) => String(d.id) === String(scope.value))) scope.value = "";
  } catch (e) { error.value = e.message + " Check that the API and database are running."; }
}
function chooseFile(event) { file.value = event.target.files?.[0] || null; notice.value = ""; }
async function upload() {
  if (!file.value || busy.value) return;
  busy.value = "upload"; error.value = ""; notice.value = "";
  const data = new FormData(); data.append("file", file.value);
  try {
    const uploaded = await api("/documents/upload", { method: "POST", body: data });
    notice.value = "Indexed " + uploaded.filename + " into " + uploaded.chunks + " chunks.";
    file.value = null;
    const input = document.querySelector("#pdf");
    if (input) input.value = "";
    await refresh();
    const doc = documents.value.find((d) => d.filename === uploaded.filename);
    if (doc) scope.value = String(doc.id);
  } catch (e) { error.value = e.message; }
  finally { busy.value = ""; }
}
async function ask() {
  if (!canAsk.value) return;
  busy.value = "query"; error.value = ""; result.value = null;
  try {
    result.value = await api("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.value.trim(), document_id: scope.value ? Number(scope.value) : null })
    });
  } catch (e) { error.value = e.message; }
  finally { busy.value = ""; }
}
async function remove(doc) {
  if (busy.value || !window.confirm("Delete " + doc.filename + " and its indexed chunks?")) return;
  busy.value = "delete"; error.value = ""; notice.value = "";
  try {
    const response = await api("/documents/" + doc.id, { method: "DELETE" });
    notice.value = response.message; result.value = null; await refresh();
  } catch (e) { error.value = e.message; }
  finally { busy.value = ""; }
}
onMounted(refresh);
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <a class="brand" href="/"><span class="brand-mark">R</span> Research<span class="brand-accent">Desk</span></a>
      <a :href="API + '/docs'" target="_blank" rel="noreferrer">API docs ↗</a>
    </header>
    <section class="hero">
      <p class="eyebrow">YOUR DOCUMENTS. GROUNDED ANSWERS.</p>
      <h1>Research with <span>evidence.</span></h1>
      <p>Upload a paper, ask a question, and explore answers supported by passages from your own documents.</p>
      <div class="pipeline"><span>PDF ingestion</span>→<span>Hybrid search</span>→<span>Reranking</span>→<span>Cited answer</span></div>
    </section>

    <div class="workspace">
      <aside class="panel">
        <div class="panel-head"><div><p class="eyebrow">LIBRARY</p><h2>Your documents</h2></div><button class="icon-button" @click="refresh" title="Refresh">↻</button></div>
        <label class="upload" for="pdf"><b>＋</b><strong>Choose a PDF</strong><small>PDF files · max 25 MB</small><input id="pdf" type="file" accept="application/pdf,.pdf" @change="chooseFile" /></label>
        <p v-if="file" class="picked">{{ file.name }} <button class="link-button" @click="file = null">Clear</button></p>
        <button class="primary wide" :disabled="!file || !!busy" @click="upload">{{ busy === 'upload' ? "Extracting and indexing…" : "Upload and index" }}</button>
        <div class="list-head"><span>Indexed documents</span><span>{{ documents.length }}</span></div>
        <p v-if="!documents.length" class="muted">Your library is empty. Upload a research PDF to get started.</p>
        <article v-for="doc in documents" :key="doc.id" class="doc-row" :class="{ active: String(doc.id) === String(scope) }">
          <label class="doc-choice"><input v-model="scope" type="radio" name="scope" :value="String(doc.id)" /><span class="pdf">PDF</span><span class="doc-meta"><b :title="doc.filename">{{ doc.filename }}</b><small>{{ doc.chunk_count }} searchable chunks</small></span></label>
          <button class="remove" :disabled="!!busy" :aria-label="'Delete ' + doc.filename" @click="remove(doc)">×</button>
        </article>
        <button v-if="scope" class="link-button clear-scope" @click="scope = ''">Search all documents</button>
      </aside>

      <section class="panel question-panel">
        <p class="eyebrow">ASK YOUR RESEARCH</p><h2>What would you like to understand?</h2>
        <p class="muted">{{ scope ? "Searching within the selected document." : "Searching across your entire library." }}</p>
        <form class="question-box" @submit.prevent="ask">
          <label class="sr-only" for="question">Research question</label>
          <textarea id="question" v-model="question" rows="4" placeholder="Ask a specific question about your papers…" />
          <div class="form-foot"><small>Answers are based on retrieved document evidence.</small><button class="primary" type="submit" :disabled="!canAsk">{{ busy === 'query' ? "Searching…" : "Ask question ↗" }}</button></div>
        </form>
        <p v-if="error" class="message error" role="alert">{{ error }}</p><p v-if="notice" class="message success" role="status">{{ notice }}</p>
        <div v-if="busy === 'query'" class="loading">Searching, reranking, and preparing an answer…</div>
        <section v-if="result" class="answer">
          <div class="answer-head"><div><p class="eyebrow">GENERATED RESPONSE</p><h3>Answer</h3></div><span class="pill">{{ result.sources.length }} sources</span></div>
          <p class="muted">{{ result.question }}</p><div class="answer-text">{{ result.answer }}</div>
          <div class="sources-head"><h3>Supporting passages</h3><small>Verify the original context</small></div>
          <article v-for="(source, i) in result.sources" :key="source.filename + source.page_number + source.chunk_index" class="source">
            <div><b>SOURCE {{ i + 1 }}</b><small>{{ source.filename }} · p. {{ source.page_number }} · chunk {{ source.chunk_index }}</small></div>
            <p>{{ source.text }}</p>
          </article>
        </section>
        <div v-else-if="busy !== 'query'" class="placeholder"><span>⌕</span><b>Your answer will appear here</b><p>Choose a document or search the library, then ask a question to see the answer and source passages.</p></div>
      </section>
    </div>
    <footer><span>AI Research Assistant</span><span>Semantic retrieval · BM25 · RRF · Cross-encoder reranking</span></footer>
  </main>
</template>
