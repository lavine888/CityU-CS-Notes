#!/usr/bin/env node

/**
 * Keep the notes repository internally consistent. Runs offline, no network.
 *
 *  1. Every directory under `courses/` is listed in the root README index.
 *  2. Every `courses/...` link in the root README points to a real directory.
 *  3. Every relative markdown link/image in the repository resolves to a file
 *     or directory that actually exists.
 *
 * The official-materials folders are intentionally partial, so links into
 * `materials/` are checked like any other relative link.
 */

import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const COURSES = join(ROOT, "courses");
const SKIP_DIRS = new Set([".git", "node_modules"]);
const EXTERNAL = /^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i;

const errors = [];

function markdownFiles(dir) {  const out = [];
  for (const name of readdirSync(dir)) {
    if (SKIP_DIRS.has(name)) continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...markdownFiles(path));
    else if (name.toLowerCase().endsWith(".md")) out.push(path);
  }
  return out;
}

function withoutCodeBlocks(text) {
  // Fenced blocks often contain illustrative links that are not meant to resolve.
  return text.replace(/```[\s\S]*?```/g, "").replace(/~~~[\s\S]*?~~~/g, "");
}

function relativeLinkTargets(text) {
  text = withoutCodeBlocks(text);
  const targets = [];
  // [label](target) and ![alt](target)
  for (const match of text.matchAll(/!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)) {
    targets.push(match[1]);
  }
  // <a href="target">
  for (const match of text.matchAll(/<a\s[^>]*href="([^"]+)"/gi)) {
    targets.push(match[1]);
  }
  return targets;
}

// 1 + 2: course index vs. actual directories.
const courseDirs = readdirSync(COURSES).filter((name) => {
  const path = join(COURSES, name);
  return statSync(path).isDirectory() && !name.startsWith(".");
});

const rootReadme = readFileSync(join(ROOT, "README.md"), "utf8");
const indexed = new Set();
for (const target of relativeLinkTargets(rootReadme)) {
  if (!target.startsWith("courses/")) continue;
  indexed.add(target.replace(/[#?].*$/, "").replace(/\/+$/, ""));
}

for (const name of courseDirs) {
  const rel = `courses/${name}`;
  if (!indexed.has(rel)) errors.push(`README.md: course directory is not in the index -> ${rel}`);
}
for (const rel of indexed) {
  if (!existsSync(join(ROOT, rel))) errors.push(`README.md: index points to a missing directory -> ${rel}`);
}

// 3: every relative link resolves.
const files = markdownFiles(ROOT);
let linkCount = 0;
for (const file of files) {
  const text = readFileSync(file, "utf8");
  for (const raw of relativeLinkTargets(text)) {
    if (EXTERNAL.test(raw)) continue;
    const target = raw.replace(/[#?].*$/, "");
    if (!target) continue;
    linkCount += 1;
    let decoded = target;
    try {
      decoded = decodeURIComponent(target);
    } catch {
      // keep the raw target if it is not valid percent-encoding
    }
    if (!existsSync(resolve(dirname(file), decoded))) {
      errors.push(`${relative(ROOT, file).replaceAll("\\", "/")}: broken relative link -> ${raw}`);
    }
  }
}

if (errors.length > 0) {
  console.error(`${errors.length} consistency problem(s):\n`);
  for (const error of errors) console.error(`  - ${error}`);
  process.exit(1);
}

console.log(
  `OK: ${courseDirs.length} course directories indexed, ${linkCount} relative links across ${files.length} markdown files resolve.`,
);
