library(readr)   # CSV 読み込み
library(dplyr)   # データ操作
library(tidyr)   # データ整形
library(ggplot2) # グラフ描画

pro_name <- "RWMOP30"
algo <- "CDP"

# === 1. CSVファイルを読み込み ===
df <- read_csv(paste0("/Users/azumayuki/Documents/LONs/feasible_ratio/", pro_name, "_", algo, ".csv"))

# === 2. 世代ごとに平均を取る ===
df_avg <- df %>%
  group_by(gen) %>%
  summarise(across(everything(), mean, na.rm = TRUE), .groups = "drop")

# === 3. 制約列を縦持ちに変換 ===
df_long <- df_avg %>%
  pivot_longer(cols = starts_with("con"),
               names_to = "constraint",
               values_to = "rate")

p <- ggplot() +
  # 各制約の実行可能率（破線）
  geom_line(data = df_long,
            aes(x = gen, y = rate, color = constraint)) +
  # 全体の実行可能率（黒線）
  geom_line(data = df_avg,
            aes(x = gen, y = all),
            color = "red", size = 1) +
  theme_minimal(base_size = 14) +
  labs(
       x = "Generation", y = "Feasible ratio") +
  theme(legend.position = "bottom")

p + ylim(0, 1) + xlim(0, 1000)