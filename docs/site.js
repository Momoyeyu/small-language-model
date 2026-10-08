(() => {
  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const inline = (text) =>
    escapeHtml(text)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^ )]+|#[a-zA-Z0-9_/-]+|[./a-zA-Z0-9_-]+)\)/g,
        (_, label, url) => `<a href="${url}" rel="noopener noreferrer">${label}</a>`);

  function markdown(source) {
    const lines = source.replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let paragraph = [], list = "", inCode = false, inTable = false, code = [];
    const flush = () => {
      if (paragraph.length) { out.push(`<p>${inline(paragraph.join(" "))}</p>`); paragraph = []; }
      if (list) { out.push(`</${list}>`); list = ""; }
      if (inTable) { out.push("</table></div>"); inTable = false; }
    };
    for (const line of lines) {
      if (line.startsWith("```")) {
        flush();
        if (inCode) { out.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`); code = []; }
        inCode = !inCode;
        continue;
      }
      if (inCode) { code.push(line); continue; }
      if (line.startsWith("|") && line.endsWith("|")) {
        if (!inTable) { flush(); out.push('<div class="table-wrap"><table>'); inTable = true; }
        if (/^\|[\s:|-]+\|$/.test(line)) continue;
        const cells = line.slice(1, -1).split("|").map((cell) => `<td>${inline(cell.trim())}</td>`).join("");
        out.push(`<tr>${cells}</tr>`);
        continue;
      }
      if (inTable) flush();
      if (!line.trim()) { flush(); continue; }
      const heading = /^(#{1,3})\s+(.+)$/.exec(line);
      if (heading) { flush(); const level = heading[1].length; out.push(`<h${level}>${inline(heading[2])}</h${level}>`); continue; }
      const item = /^([-*]|\d+\.)\s+(.+)$/.exec(line);
      if (item) {
        if (paragraph.length) { out.push(`<p>${inline(paragraph.join(" "))}</p>`); paragraph = []; }
        const kind = item[1].endsWith(".") ? "ol" : "ul";
        if (list && list !== kind) { out.push(`</${list}>`); list = ""; }
        if (!list) { out.push(`<${kind}>`); list = kind; }
        out.push(`<li>${inline(item[2])}</li>`);
        continue;
      }
      if (list) { out.push(`</${list}>`); list = ""; }
      paragraph.push(line.trim());
    }
    flush();
    if (inCode) out.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
    return out.join("\n");
  }

  const I18N = {
    zh: { guide: "知识库", diagram: "架构图", diagramTitle: "SLM 架构总览", diagramNote: "三条消费通道与 slm 核心依赖；节点含源码证据，可在新标签页单独查看。", diagramLink: "单独打开可交互 Archify 图 ↗", notFound: "未找到该章节。请从左侧选择。", loadFail: "章节加载失败", hint: "请使用 HTTP server（如 make docs），而非 file://。", fatal: "文档无法加载", brand: "SLM / CORE ARCHITECTURE" },
    en: { guide: "Guide", diagram: "Diagram", diagramTitle: "SLM architecture overview", diagramNote: "Three consumption lanes and core slm dependencies. Nodes carry source evidence; open in a new tab for the full view.", diagramLink: "Open the interactive Archify diagram ↗", notFound: "Chapter not found. Pick one on the left.", loadFail: "Failed to load chapter", hint: "Serve over HTTP (e.g. make docs); file:// is not supported.", fatal: "Documentation failed to load", brand: "SLM / CORE ARCHITECTURE" },
  };

  const els = {
    sidebar: document.getElementById("sidebar"),
    content: document.getElementById("content"),
    revision: document.getElementById("revision"),
    guide: document.getElementById("guideLink"),
    diagram: document.getElementById("diagramLink"),
    zh: document.getElementById("zhLink"),
    en: document.getElementById("enLink"),
    brand: document.getElementById("brand"),
  };

  let manifest;

  const route = () => {
    const parts = decodeURIComponent(location.hash.slice(1)).split("/").filter(Boolean);
    const lang = parts[0] === "en" ? "en" : "zh";
    const id = parts[0] === "en" ? parts[1] || "overview" : parts[0] || "overview";
    return { lang, id };
  };

  const hrefFor = (lang, id) => (lang === "en" ? `#en/${id}` : `#${id}`);

  const t = () => I18N[route().lang];

  function buildSidebar() {
    const { lang, id } = route();
    const langData = manifest.languages[lang];
    els.sidebar.innerHTML = langData.sections
      .map(
        (section) =>
          `<h2>${escapeHtml(section.title)}</h2>` +
          section.chapters
            .map((ch) => `<a data-id="${escapeHtml(ch.id)}" href="${hrefFor(lang, ch.id)}"${ch.id === id ? ' class="active"' : ""}>${escapeHtml(ch.title)}</a>`)
            .join("")
      )
      .join("") + `<div class="side-foot">SOURCE · ${escapeHtml(manifest.sourceRevision.slice(0, 12))}</div>`;
  }

  async function render() {
    const { lang, id } = route();
    const copy = t();
    const chapters = manifest.languages[lang].sections.flatMap((s) => s.chapters);
    const chapter = chapters.find((c) => c.id === id);
    document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
    document.title = `${manifest.title[lang]} · ${chapter ? chapter.title : copy.diagramTitle}`;
    els.brand.textContent = copy.brand;
    els.guide.textContent = copy.guide;
    els.diagram.textContent = copy.diagram;
    els.guide.href = hrefFor(lang, chapter ? chapter.id : "overview");
    els.diagram.href = hrefFor(lang, "diagram");
    for (const language of ["zh", "en"]) {
      const link = els[language];
      link.href = hrefFor(language, id);
      if (language === lang) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    els.guide.classList.toggle("active", id !== "diagram");
    els.diagram.classList.toggle("active", id === "diagram");
    buildSidebar();
    if (id === "diagram") {
      els.content.innerHTML = `<div class="hero"><h1>${copy.diagramTitle}</h1><p>${copy.diagramNote}</p><p><a class="diagram-link" href="architecture.html" target="_blank" rel="noopener">${copy.diagramLink}</a></p><iframe class="diagram" title="SLM architecture diagram" src="architecture.html"></iframe></div>`;
      window.scrollTo({ top: 0 });
      return;
    }
    if (!chapter) {
      els.content.innerHTML = `<p class="error">${copy.notFound}</p>`;
      return;
    }
    try {
      const response = await fetch(chapter.file);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      els.content.innerHTML = `<article class="prose"><div class="crumb">SLM / ${escapeHtml(chapter.title)}</div>${markdown(await response.text())}</article>`;
    } catch (err) {
      els.content.innerHTML = `<p class="error">${copy.loadFail}：${escapeHtml(err.message)}。${copy.hint}</p>`;
    }
    window.scrollTo({ top: 0 });
  }

  async function load() {
    const response = await fetch("manifest.json");
    if (!response.ok) throw new Error(`manifest: HTTP ${response.status}`);
    manifest = await response.json();
    els.revision.textContent = `SOURCE · ${manifest.sourceRevision.slice(0, 8)}`;
    window.addEventListener("hashchange", render);
    await render();
  }

  load().catch((err) => {
    els.content.innerHTML = `<p class="error">${t().fatal}：${escapeHtml(err.message)}。${t().hint}</p>`;
  });
})();
