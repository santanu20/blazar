// Client-side search over the Jekyll-generated index (search.json).
// Hand-rolled: ~17 pages, no external engine needed. Scoring favors title
// matches, then earliest and densest hits in the body text.
(function () {
  "use strict";

  var INDEX = null;

  function qs(q) { return (q || "").trim().toLowerCase(); }

  function score(doc, terms) {
    var title = (doc.title || "").toLowerCase();
    var body = (doc.text || "").toLowerCase();
    var total = 0;
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i];
      if (!t) continue;
      var inTitle = title.indexOf(t) !== -1;
      var first = body.indexOf(t);
      if (!inTitle && first === -1) return 0; // every term must appear
      var s = inTitle ? 40 : 0;
      if (first !== -1) {
        s += Math.max(4, 20 - Math.floor(first / 400)); // earlier = better
        var count = body.split(t).length - 1;
        s += Math.min(count, 10); // denser = better, capped
      }
      total += s;
    }
    return total;
  }

  function snippetFor(doc, terms) {
    var body = doc.text || "";
    var lower = body.toLowerCase();
    var at = -1;
    for (var i = 0; i < terms.length; i++) {
      at = lower.indexOf(terms[i]);
      if (at !== -1) break;
    }
    if (at === -1) return body.slice(0, 160);
    var start = Math.max(0, at - 70);
    var raw = body.slice(start, start + 220);
    var esc = raw.replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
    for (var j = 0; j < terms.length; j++) {
      esc = esc.replace(
        new RegExp(terms[j].replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"),
        "<mark>$&</mark>"
      );
    }
    return (start > 0 ? "…" : "") + esc + "…";
  }

  function renderResults(box, hits, terms) {
    box.innerHTML = "";
    hits.forEach(function (h) {
      var div = document.createElement("div");
      div.className = "search-result";
      var a = document.createElement("a");
      a.href = h.doc.url;
      a.textContent = h.doc.title || h.doc.url;
      var url = document.createElement("span");
      url.className = "url";
      url.textContent = h.doc.url;
      var p = document.createElement("p");
      p.innerHTML = snippetFor(h.doc, terms);
      a.appendChild(url);
      div.appendChild(a);
      div.appendChild(p);
      box.appendChild(div);
    });
  }

  function run(input, box, countEl) {
    var q = qs(input.value);
    if (!INDEX) return;
    if (q.length < 2) {
      box.innerHTML = "";
      if (countEl) countEl.textContent = "";
      return;
    }
    var terms = q.split(/\s+/);
    var hits = [];
    for (var i = 0; i < INDEX.length; i++) {
      var s = score(INDEX[i], terms);
      if (s > 0) hits.push({ doc: INDEX[i], s: s });
    }
    hits.sort(function (a, b) { return b.s - a.s; });
    hits = hits.slice(0, 12);
    renderResults(box, hits, terms);
    if (countEl) {
      countEl.textContent = hits.length
        ? hits.length + " page" + (hits.length === 1 ? "" : "s")
        : "no matches — try fewer terms";
    }
  }

  function boot() {
    var input = document.getElementById("search-input");
    var box = document.getElementById("search-results");
    if (!input || !box) {
      // Not on the search page: wire the nav shortcut ("/" focuses search).
      document.addEventListener("keydown", function (e) {
        if (e.key === "/" && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
          var open = document.querySelector("[data-search-open]");
          if (open) { e.preventDefault(); open.click(); }
        }
      });
      return;
    }
    var countEl = document.getElementById("search-count");
    var timer = null;
    input.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(function () { run(input, box, countEl); }, 90);
    });
    var req = new XMLHttpRequest();
    req.open("GET", document.documentElement.getAttribute("data-search-index") || "/blazar/search.json");
    req.onload = function () {
      try { INDEX = JSON.parse(req.responseText); } catch (e) { INDEX = []; }
      run(input, box, countEl);
    };
    req.send();
    var prefill = new URLSearchParams(window.location.search).get("q");
    if (prefill) { input.value = prefill; run(input, box, countEl); }
    input.focus();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
