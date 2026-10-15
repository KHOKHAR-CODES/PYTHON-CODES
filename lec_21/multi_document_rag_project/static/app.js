const form = document.querySelector("#ask-form");
const question = document.querySelector("#question");
const button = document.querySelector("#ask-button");
const result = document.querySelector("#result");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const sourceCount = document.querySelector("#source-count");
const loading = document.querySelector("#loading");
const error = document.querySelector("#error");

async function ask() {
  const value = question.value.trim();
  if (!value) return;
  loading.hidden = false;
  error.hidden = true;
  result.hidden = true;
  button.disabled = true;
  button.querySelector("span").textContent = "SEARCHING...";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong.");
    answer.textContent = data.answer;
    sources.replaceChildren(
      ...(data.sources || []).map((source) => {
        const item = document.createElement("div");
        item.className = "source";
        item.textContent = source;
        return item;
      }),
    );
    sourceCount.textContent = `(${(data.sources || []).length})`;
    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
  } finally {
    loading.hidden = true;
    button.disabled = false;
    button.querySelector("span").textContent = "SEARCH DOCUMENTS";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask();
});
question.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") ask();
});
