# Security — 凭证安全

## 绝不提交凭证

本仓库**绝不**接受任何密码、token、API key、私钥等凭证。

- 不要把 GitHub 密码、Personal Access Token 写进任何文件、脚本、git 配置或 commit message。
- 不要把模型下载 token、HF token、云厂商 secret 写进代码。
- `.gitignore` 已屏蔽 `.env`、`*.secret`、`credentials*`、`netrc`。

## 推送仓库的正确方式

用 GitHub Personal Access Token (PAT)，且**只通过 git 凭证助手或交互输入**，
不要把 token 明文写进 remote URL（它会落进 `.git/config` 明文）。

推荐：用 git credential helper 缓存，或推送时交互输入 token。

## 如果你不慎提交了凭证

立即在 GitHub 撤销该 token / 修改密码，然后用 `git filter-repo` 或 BFG
清除历史中的凭证，再 force push。仅删除文件不够——git 历史仍保留。
