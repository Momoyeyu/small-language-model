# 边界与静态发布

教学实现的边界是架构的一部分。

## 模型边界

- 默认 Transformer 不会因新增 MoE、RoPE 或 QK-Norm 自动升级。
- 它是 Encoder–Decoder，不是 decoder-only production LLM。
- 没有 KV cache、并行通信或 fused kernels。

## 实验边界

- CartPole cutoff 使用有限时域终止语义。
- T5、Shaw、ALiBi 非 2 次幂等细节按文档所述简化。
- PI/NTK 数值展示不等于测得长上下文提升。

`eval_ppl` 对 batch 每行除首 token 外的 next-token target 恰好计分一次，要求 `stride < ctx`，并恢复原始 `model.training`。TinyLM 使用不同 BOS 对齐，未经 adapter 不应直接传入。

## 文档运行与 GitHub Pages

本站是纯静态文件：`index.html` 壳 + `manifest.json` 章节清单 + `chapters/` Markdown + Archify 生成的 `architecture.html`。无构建步骤、无 CDN、无外部字体、无秘密信息；章节内容由浏览器内的轻量 Markdown 渲染器装配，禁用 JavaScript 时请直接阅读 `chapters/` 下的 Markdown 源文件。

```bash
make docs              # 本地预览，默认 127.0.0.1:8000，可用 PORT 覆盖
make check-docs        # 校验清单、链接、双语一致性与核心 import 边
```

GitHub Pages 从 `master` 分支的 `/docs` 目录发布，`.nojekyll` 保持静态资源原样。发布前应确认公开目录不含凭据；推送 `master` 中的 `docs/` 改动会自动更新站点。

## 架构图维护

`architecture.html` 由 [archify](https://github.com/tt-a1i/archify) 从 `candidate.json` 生成，节点证据钉在 manifest 记录的源码 revision 上。核心结构变化时先更新 `candidate.json` 的组件、边和 `sources` 行号，再重新生成：

```bash
node archify.mjs finalize architecture docs/candidate.json docs/architecture.html \
  --repo-root . --quality showcase --json
```
