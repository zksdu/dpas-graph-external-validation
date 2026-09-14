#!/usr/bin/env Rscript
# 导出 Visium RDS 的完整 meta.data（聚类、区域注释、counts）用于下游分析
suppressPackageStartupMessages({
  library(Seurat)
})

out_dir <- "data/pe_placenta/export"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

for (sec in c("A", "B")) {
  rds <- sprintf("data/pe_placenta/section%s_visium_spotlight.rds", sec)
  cat("===== reading", rds, "=====\n")
  obj <- readRDS(rds)
  md <- obj@meta.data
  md$barcode <- rownames(md)
  # 坐标
  img <- obj@images[[1]]
  if (!is.null(img@coordinates)) {
    coords <- img@coordinates
    # VisiumV1: 行名为 barcode，列含 imagerow/imagecol
    common <- intersect(md$barcode, rownames(coords))
    md$imagerow <- NA_real_; md$imagecol <- NA_real_
    md$imagerow[match(common, md$barcode)] <- coords[common, "imagerow"]
    md$imagecol[match(common, md$barcode)] <- coords[common, "imagecol"]
  }
  f <- sprintf("%s/section%s_metadata.csv", out_dir, sec)
  write.csv(md, f, row.names = FALSE)
  cat("wrote", f, "dim:", nrow(md), "x", ncol(md), "\n")
  cat("cols:", paste(colnames(md), collapse = " | "), "\n")

  # 看一眼 res_ss 的取值分布
  if ("res_ss" %in% colnames(md)) {
    cat("res_ss summary:\n"); print(summary(md$res_ss))
    cat("res_ss table (head):\n"); print(head(sort(table(md$res_ss), decreasing = TRUE), 20))
  }
  for (cl in c("seurat_clusters", "SCT_snn_res.2", "SCT_snn_res.1")) {
    if (cl %in% colnames(md)) {
      cat(cl, "levels:", length(unique(md[[cl]])), "\n")
      print(head(sort(table(md[[cl]]), decreasing = TRUE), 15))
    }
  }
  cat("\n")
}
cat("DONE\n")
