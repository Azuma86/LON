# ---------------------------------------------------------
# 必要なライブラリをロード
# ---------------------------------------------------------
library(dplyr)
library(tidyr)
library(igraph)      # グラフ作成・操作用
library(ggplot2)     # 可視化用
library(scales)      # ログスケール表示など
library(Matrix)
library(Rlinsolve)

# ---------------------------------------------------------
# 1. データ読み込みと前処理
# ---------------------------------------------------------
problem_name <- "RWMOP22"

# domain_info.csv を読み込んで、その中から該当問題行を取得
domain_df <- read.csv("/Users/azumayuki/Documents/LONs/domain_info.csv", stringsAsFactors=FALSE)
row_domain <- domain_df %>% filter(problem == problem_name) %>% slice(1)

# lower, upper を数値ベクトル化
lower <- as.numeric(unlist(strsplit(row_domain$lower, ",")))
upper <- as.numeric(unlist(strsplit(row_domain$upper, ",")))
diff <- upper - lower

# ローカルサーチ結果などが入った CSV を読み込む
# ファイルパスは実際の環境に合わせて修正してください。
data_raw <- read.csv(file.path("/Users/azumayuki/Documents/LONs/data09-20-pre", paste0("local_search", problem_name, ".csv")),
                     stringsAsFactors=FALSE)

# 制約列 (Con_*) を探し合計CVを計算
con_cols <- grep("^Con_", colnames(data_raw), value=TRUE)
data_raw <- data_raw %>%
  rowwise() %>%
  mutate(CV = sum(pmax(0, c_across(all_of(con_cols))))) %>%
  ungroup()

# 世代 (Gen) とID (ID) で並べ替え
data_sorted <- data_raw %>% arrange(ID, Gen)

# ---------------------------------------------------------
# 2. グラフ作成 (igraph) 
# ---------------------------------------------------------
# まずノード情報を作る
#   項目: Gen, ID, X(ベクトル), CV
#   ノードの名前は行インデックスにしておく
# ---------------------------------------------------------
# X列をまとめる
x_cols <- grep("^X_", colnames(data_sorted), value=TRUE)

nodes_df <- data_sorted %>%
  mutate(idx = row_number()) %>%
  select(idx, ID, Gen, CV, all_of(x_cols)) %>%
  as.data.frame()  # <- as.data.frame()を挟む

rownames(nodes_df) <- nodes_df$idx

# ---------------------------------------------------------
# 3. エッジの作成
#    同一ID かつ Genが連続 (g+1=次世代) の場合にエッジを張る
# ---------------------------------------------------------
edge_list <- list()
prev_row <- NULL

for (i in seq_len(nrow(nodes_df))) {
  current_row <- nodes_df[i, ]
  if (!is.null(prev_row)) {
    # 同じIDで, 世代が1つ上がっていたらエッジ
    if (prev_row$ID == current_row$ID && 
        current_row$Gen == prev_row$Gen + 1) {
      edge_list <- append(edge_list, list(c(prev_row$idx, current_row$idx)))
    }
  }
  prev_row <- current_row
}

edges_df <- do.call(rbind, edge_list)
colnames(edges_df) <- c("from", "to")
edges_df <- as.data.frame(edges_df)

# いったんグラフ構築
g <- graph_from_data_frame(d = edges_df, 
                           vertices = nodes_df,
                           directed = TRUE)

# ---------------------------------------------------------
# 4. 重複ノードの統合 (まったく同じ X ベクトルを持つノード)
# ---------------------------------------------------------
#   - 同じ設計変数 X のノードを1つの代表ノードにまとめる
#   - 代表以外のノードに入っていたエッジは代表にリダイレクトし、ノードを削除
# ---------------------------------------------------------

# X を文字列化して重複判定
paste_x_values <- function(rowX) {
  paste(format(rowX, digits=15), collapse=",")
}
print(x_cols)
# 各ノードの X ベクトルを文字列に
x_strings <- apply(as.matrix(nodes_df[x_cols]), 1, paste_x_values)

# 同じ文字列を持つノード達をグループ化
x_groups <- split(names(x_strings), f=x_strings)  # names(...) は rownames(nodes_df) = idx

# 重複ノードを1つの代表にまとめる
for (group_key in names(x_groups)) {
  group_nodes <- x_groups[[group_key]]
  if (length(group_nodes) > 1) {
    # 代表ノード(先頭)を決める
    rep_node <- group_nodes[1]
    dup_nodes <- group_nodes[-1]
    # 重複ノードの predecessor を代表にリダイレクト
    for (dn in dup_nodes) {
      preds <- igraph::neighbors(g, dn, mode="in")
      for (p in preds) {
        if (p$name != rep_node && 
            !are.connected(g, p$name, rep_node)) {
          g <- add_edges(g, c(p$name, rep_node))
        }
      }
      # 重複ノードの successor を代表にリダイレクト
      succs <- igraph::neighbors(g, dn, mode="out")
      for (s in succs) {
        if (s$name != rep_node && 
            !are.connected(g, rep_node, s$name)) {
          g <- add_edges(g, c(rep_node, s$name))
        }
      }
      # 重複ノード削除
      g <- delete_vertices(g, dn)
    }
  }
}

stress_majorization_1d <- function(dist_matrix, max_iter = 1000, tol = 1e-4) {
  N <- nrow(dist_matrix)
  epsilon <- 1e-10
  
  # 0距離を回避
  dist_matrix[dist_matrix < epsilon] <- epsilon
  
  # W = 1 / dist_matrix
  W <- 1 / dist_matrix
  
  # Lw = diag(rowSums(W)) - W
  rowSumsW <- rowSums(W)
  Lw <- diag(rowSumsW) - W
  
  # 疎行列化 (共役勾配法のA行列)
  Lw_sp <- Matrix(Lw, sparse=TRUE)
  
  # 初期解
  x <- runif(N)
  x <- x - mean(x)
  
  stress_prev <- Inf
  
  for (iter in seq_len(max_iter)) {
    # δ = W * dist_matrix
    delta <- W * dist_matrix
    
    # diff[i,j] = x[i] - x[j]
    diff <- outer(x, x, FUN = "-")
    norm_val <- pmax(abs(diff), epsilon)
    
    # B = delta * diff / norm_val
    B <- delta * diff / norm_val
    row_sum_B <- rowSums(B)
    Lz <- diag(row_sum_B) - B
    
    # 右辺
    rhs <- Lz %*% x
    init_vec <- rep(Inf, length(rhs))
    # ---- CGで解く ----
    sol <- lsolve.cg(A = Lw_sp, B = rhs, xinit = init_vec, reltol = 1e-08, maxiter = 10000)
    x_new <- sol$x
    
    # 平均0に
    x_new <- x_new - mean(x_new)
    
    # stress 計算
    new_diff <- outer(x_new, x_new, FUN = "-")
    norm_diff <- abs(new_diff)
    stress <- 0.5 * sum(W * (norm_diff - dist_matrix)^2)
    
    rel_change <- (stress_prev - stress) / stress_prev
    if (!is.na(rel_change) && rel_change < tol) {
      cat(sprintf("Converged after %d iterations, stress = %.4e\n", iter, stress))
      return(x_new)
    }
    
    x <- x_new
    stress_prev <- stress
  }
  
  cat(sprintf("Max iterations reached, final stress = %.4e\n", stress_prev))
  return(x)
}

# ---------------------------------------------------------
# グラフのノード情報を再取得 (重複削除後)
# ---------------------------------------------------------
nodes_now <- V(g)  # igraphの頂点オブジェクト
N <- length(nodes_now)

# ノードの属性をデータフレーム化
g_nodes_df <- data.frame(
  idx = V(g)$name,    
  ID  = V(g)$ID,
  Gen = V(g)$Gen,
  CV  = V(g)$CV,
  stringsAsFactors = FALSE
)

# Xベクトルを行列としてまとめる
#   重複削除の段階で消えたものはないので再抽出
nodeX_list <- lapply(nodes_now, function(v) {
  unlist(v[x_cols], use.names=FALSE)
})
X_mat <- do.call(rbind, nodeX_list)
# 正規化
X_norm <- t( (t(X_mat) - lower) / diff )

# 距離行列 (ユークリッド)
dist_matrix <- as.matrix(dist(X_norm, method="euclidean"))

# ストレス・メジャライゼーションで1D座標を得る
x_coords <- stress_majorization_1d(dist_matrix)

# ノード順序が混乱しないよう、順番を合わせる
g_nodes_df$x_coord <- x_coords


# ---------------------------------------------------------
# 6. 可視化
#   - x_coord(1D埋め込み) を x軸
#   - CV を y軸
#   - エッジは ggraph または手動で描画可
#   - ここでは簡易的に ggplot + geom_segment でエッジを描いてみる例
# ---------------------------------------------------------
# エッジデータフレーム
edges_now <- as_data_frame(g, what="edges")
colnames(edges_now) <- c("from", "to")

# from/to それぞれの座標を結合
edges_plot_df <- edges_now %>%
  left_join(g_nodes_df, by=c("from"="idx")) %>%
  rename(x_from = x_coord, CV_from = CV) %>%
  left_join(g_nodes_df, by=c("to"="idx")) %>%
  rename(x_to = x_coord, CV_to = CV)

# ノードを sinkかどうか（outdegree=0かどうか）で分類
outdeg <- degree(g, mode="out")
sink_nodes <- names(which(outdeg == 0))
mid_nodes <- setdiff(g_nodes_df$idx, sink_nodes)

# sink かつ CV=0 / CV>0
final_feasible_idx   <- sink_nodes[g_nodes_df$CV[match(sink_nodes, g_nodes_df$idx)] == 0]
final_infeasible_idx <- sink_nodes[g_nodes_df$CV[match(sink_nodes, g_nodes_df$idx)] > 0]

# 中間ノード
mid_feasible_idx   <- mid_nodes[g_nodes_df$CV[match(mid_nodes, g_nodes_df$idx)] == 0]
mid_infeasible_idx <- mid_nodes[g_nodes_df$CV[match(mid_nodes, g_nodes_df$idx)] > 0]

g_nodes_df$category <- "mid_infeasible"
g_nodes_df$category[g_nodes_df$idx %in% mid_feasible_idx]   <- "mid_feasible"
g_nodes_df$category[g_nodes_df$idx %in% final_feasible_idx] <- "final_feasible"
g_nodes_df$category[g_nodes_df$idx %in% final_infeasible_idx] <- "final_infeasible"

# カテゴリごとに色・サイズを設定する例
cat_colors <- c("mid_infeasible"   = "skyblue",
                "mid_feasible"     = "salmon",
                "final_feasible"   = "red",
                "final_infeasible" = "blue")
cat_sizes  <- c("mid_infeasible"   = 1.5,
                "mid_feasible"     = 1.5,
                "final_feasible"   = 3.5,
                "final_infeasible" = 3.5)

# symlog的なスケールにするためには ggplot で pseudo_log_trans などを使用
# (ただし“symlog”を完全に再現するには工夫が要ります)
ggplot() +
  # エッジを矢印付きで描くなら、arrowパラメータを使ってgeom_segment
  geom_segment(
    data = edges_plot_df,
    aes(x = x_from, y = CV_from, xend = x_to, yend = CV_to),
    arrow = arrow(length=unit(0.2,"cm")),
    size = 0.5, alpha=0.3, color="gray"
  ) +
  # ノード散布図
  geom_point(
    data = g_nodes_df,
    aes(x = x_coord, y = CV, color = category, size=category),
    shape = 21, stroke=1
  ) +
  scale_color_manual(values=cat_colors) +
  scale_size_manual(values=cat_sizes) +
  # CV軸を対数スケールっぽく
  scale_y_continuous(trans = scales::pseudo_log_trans(base = 10, sigma=1e-6),
                     limits = c(-1e-6, 1e5)) +
  theme_minimal(base_size=14) +
  labs(x="1D embedding (stress majorization)", y="CV") +
  ggtitle("Constraint Violation vs 1D Embedding")