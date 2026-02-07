#!/usr/bin/env node

import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { THEMES, renderMermaid } from "beautiful-mermaid";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");
const SOURCE_ROOT = path.join(REPO_ROOT, "docs", "diagrams", "src");
const OUTPUT_ROOT = path.join(REPO_ROOT, "docs", "assets", "diagrams", "generated");

const DEFAULT_THEME = "github-light";
const DEFAULT_FONT = "Inter";
const DEFAULT_PADDING = 32;

const args = new Set(process.argv.slice(2));
const validArgs = new Set(["--check"]);
for (const arg of args) {
  if (!validArgs.has(arg)) {
    console.error(`Unsupported argument: ${arg}`);
    console.error("Usage: node scripts/render_diagrams.mjs [--check]");
    process.exit(2);
  }
}
const checkOnly = args.has("--check");

async function pathExists(targetPath) {
  try {
    await readFile(targetPath);
    return true;
  } catch (error) {
    if (error && error.code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

async function listFilesRecursive(rootDir, ext) {
  const files = [];

  async function walk(currentDir) {
    let entries;
    try {
      entries = await readdir(currentDir, { withFileTypes: true });
    } catch (error) {
      if (error && error.code === "ENOENT") {
        return;
      }
      throw error;
    }

    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        await walk(fullPath);
        continue;
      }
      if (entry.isFile() && fullPath.endsWith(ext)) {
        files.push(fullPath);
      }
    }
  }

  await walk(rootDir);
  return files;
}

function sourceToOutputPath(sourcePath) {
  const relPath = path.relative(SOURCE_ROOT, sourcePath);
  return path.join(OUTPUT_ROOT, relPath.replace(/\.mmd$/i, ".svg"));
}

async function readTextIfExists(targetPath) {
  try {
    return await readFile(targetPath, "utf8");
  } catch (error) {
    if (error && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

function sortPaths(paths) {
  return [...paths].sort((a, b) => a.localeCompare(b));
}

async function main() {
  const sourceFiles = await listFilesRecursive(SOURCE_ROOT, ".mmd");

  if (sourceFiles.length === 0) {
    console.log(`No Mermaid sources found in ${SOURCE_ROOT}`);
  }

  const renderedByOutput = new Map();
  for (const sourcePath of sourceFiles) {
    const sourceText = await readFile(sourcePath, "utf8");
    const svg = await renderMermaid(sourceText, {
      ...THEMES[DEFAULT_THEME],
      font: DEFAULT_FONT,
      padding: DEFAULT_PADDING,
    });

    const outputPath = sourceToOutputPath(sourcePath);
    renderedByOutput.set(outputPath, svg.endsWith("\n") ? svg : `${svg}\n`);
  }

  const expectedOutputs = new Set(renderedByOutput.keys());
  const existingOutputs = new Set(await listFilesRecursive(OUTPUT_ROOT, ".svg"));

  const staleOutputs = sortPaths([...existingOutputs].filter((item) => !expectedOutputs.has(item)));
  const missingOutputs = [];
  const outdatedOutputs = [];

  for (const [outputPath, expectedSvg] of renderedByOutput.entries()) {
    const currentSvg = await readTextIfExists(outputPath);
    if (currentSvg === null) {
      missingOutputs.push(outputPath);
      continue;
    }
    if (currentSvg !== expectedSvg) {
      outdatedOutputs.push(outputPath);
    }
  }

  if (checkOnly) {
    if (missingOutputs.length || outdatedOutputs.length || staleOutputs.length) {
      if (missingOutputs.length) {
        console.error("Missing generated SVGs:");
        for (const outputPath of missingOutputs) {
          console.error(`  - ${path.relative(REPO_ROOT, outputPath)}`);
        }
      }
      if (outdatedOutputs.length) {
        console.error("Outdated generated SVGs:");
        for (const outputPath of outdatedOutputs) {
          console.error(`  - ${path.relative(REPO_ROOT, outputPath)}`);
        }
      }
      if (staleOutputs.length) {
        console.error("Stale generated SVGs (no matching .mmd source):");
        for (const outputPath of staleOutputs) {
          console.error(`  - ${path.relative(REPO_ROOT, outputPath)}`);
        }
      }
      console.error("Run `npm run render-diagrams` and commit the generated SVG changes.");
      process.exit(1);
    }

    console.log(`Diagram check passed (${sourceFiles.length} source file(s)).`);
    return;
  }

  await mkdir(OUTPUT_ROOT, { recursive: true });

  for (const [outputPath, expectedSvg] of renderedByOutput.entries()) {
    await mkdir(path.dirname(outputPath), { recursive: true });
    await writeFile(outputPath, expectedSvg, "utf8");
    console.log(`Wrote ${path.relative(REPO_ROOT, outputPath)}`);
  }

  for (const stalePath of staleOutputs) {
    if (await pathExists(stalePath)) {
      await rm(stalePath);
      console.log(`Removed stale ${path.relative(REPO_ROOT, stalePath)}`);
    }
  }

  console.log(`Rendered ${sourceFiles.length} diagram(s) with theme ${DEFAULT_THEME}.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
