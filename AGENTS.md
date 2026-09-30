# 仓库说明

这是个人笔记站，用 Hugo 和 PaperMod 发布。在这里记录学习笔记，不是维护一个通用博客模板。

## 目录

- 站点配置在 `hugo.yaml`。
- 笔记正文放在 `content/posts/`。`archives.md`、`search.md` 这类工具页留在 `content/` 根目录。
- 新笔记用 `archetypes/default.md`，命令是 `hugo new posts/example-title.md`。
- 站点级覆盖放 `layouts/`，自定义样式放 `assets/css/extended/`。
- `themes/PaperMod/` 是上游主题，除非任务明确要求改主题，否则不要改。
- `public/` 是构建产物，由 GitHub Pages 部署，不要手改。

## 写笔记

使用 Markdown 和 YAML front matter。文件名小写并用连字符，例如 `content/posts/rl-dp-mc-td.md`。

每篇笔记写上 `title`、带时区的 `date`（如 `2026-09-29T18:01:04+08:00`）、`description` 和 `tags`。学科用 `categories`，细概念用 `tags`，同一本书或课程的连续笔记用 `series`。含公式时设置 `math: true`，行内用 `$...$`，独立公式用 `$$...$$`。公式块里不要让某一行单独是 `=`，也不要让行首是 `+ ` 或 `- `，否则 Markdown 会把它当成标题或列表。

未完成的笔记保持 `draft: true`。`hugo.yaml` 里 `buildDrafts: false`，草稿不会出现在正式构建中。

站点目前只有中文。`hugo.yaml` 里的 `disableLanguages: [en]` 暂时关闭英文站，需要恢复时删掉这项。无后缀的页面属于中文；英文稿仍用 `.en.md` 保存，关闭期间不会发布。不要把两种语言写进同一篇。保留作者原来的表述，不要擅自改写成博客腔，也不要大段润色。

YAML 列表和嵌套配置用两个空格缩进。

## 构建

- `hugo server -D`：本地预览，包含草稿。
- `hugo`：构建到 `public/`。
- `hugo --gc --minify`：更接近正式构建。

推送到 `main` 后，GitHub Actions（`.github/workflows/hugo.yaml`）用 Hugo Extended `0.167.0` 构建，并把 `public/` 部署到 GitHub Pages。草稿不会进正式站点。

改内容后用 `hugo server -D` 看对应笔记、导航、搜索和归档。改配置、布局或 CSS 后再跑 `hugo --gc --minify`。

## 给后续协作的约束

先读这份文件再改它，保留已有约定，不要整文件推倒重写。

整理目录、批量重命名、改信息架构之前先和用户确认。美化只改 `assets/css/extended/` 和必要的 `layouts/` 覆盖。提交保持小而集中，说明为什么改。
