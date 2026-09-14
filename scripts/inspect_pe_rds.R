# 盘点 Charité PE 胎盘 Visium spotlight RDS 结构
suppressPackageStartupMessages(library(Seurat))
out <- "D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/rds_inventory.txt"
con <- file(out, open = "wt")
sink(con, type = "output")

for (f in c("sectionA_visium_spotlight.rds", "sectionB_visium_spotlight.rds")) {
  p <- file.path("D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta", f)
  cat("=====", f, "=====\n")
  obj <- readRDS(p)
  cat("class:", class(obj), "\n")
  if (inherits(obj, "Seurat")) {
    cat("dims (genes x spots):", nrow(obj), "x", ncol(obj), "\n")
    cat("assays:", paste(Assays(obj), collapse = ", "), "\n")
    for (a in Assays(obj)) {
      cat("  assay", a, ": dim", dim(obj[[a]]), "\n")
    }
    cat("meta.data columns:", paste(colnames(obj@meta.data), collapse = ", "), "\n")
    cat("reductions:", paste(Reductions(obj), collapse = ", "), "\n")
    # 关键元数据取值
    for (col in c("orig.ident", "Condition", "condition", "group", "disease", "sample", "Sample", "ident")) {
      if (col %in% colnames(obj@meta.data)) {
        cat("  --", col, ":\n")
        print(table(obj@meta.data[[col]]))
      }
    }
    # spot spatial 坐标
    if ("spatial" %in% Reductions(obj)) {
      emb <- Embeddings(obj, "spatial")
      cat("spatial coords:", dim(emb), "range x:", range(emb[,1]), "range y:", range(emb[,2]), "\n")
    }
    # 默认 ident
    cat("default ident:\n"); print(table(Idents(obj)))
  } else if (is.list(obj)) {
    cat("list names:", paste(names(obj), collapse = ", "), "\n")
    str(obj, max.level = 2, list.len = 20)
  } else {
    str(obj, max.level = 2, list.len = 20)
  }
  cat("\n")
}

sink()
close(con)
cat("done ->", out, "\n")
