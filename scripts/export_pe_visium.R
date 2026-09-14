# 导出 PE Visium：Spatial 原始计数 + 像素坐标 + 解卷积比例（供 DPAS-Graph 迁移推理）
suppressMessages(library(Seurat))
suppressMessages(library(Matrix))
outdir <- "D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/export"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

deconv_cols <- c("vVCT", "vSCT", "vEVT", "vHBC", "vFB1", "vMC", "vTcell", "vVEC", "PAMM", "vEB1", "APAhi.tropho")

for (f in c("sectionA", "sectionB")) {
  p <- sprintf("D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/%s_visium_spotlight.rds", f)
  obj <- readRDS(p)
  cat(f, "loaded:", dim(obj)[2], "spots\n")

  # 1) Spatial 原始计数矩阵（spots x genes 转出为 genes x spots MTX）
  counts <- GetAssayData(obj, assay = "Spatial", layer = "counts")
  writeMM(as(counts, "dgCMatrix"), file.path(outdir, paste0(f, "_counts.mtx")))
  writeLines(rownames(counts), file.path(outdir, paste0(f, "_genes.txt")))
  writeLines(colnames(counts), file.path(outdir, paste0(f, "_barcodes.txt")))
  cat("  counts:", nrow(counts), "genes x", ncol(counts), "spots\n")

  # 2) 像素坐标（VisiumV1 image slot）
  img <- obj@images[[1]]
  cc <- img@coordinates  # imagecol, imagerow
  write.csv(data.frame(barcode = rownames(cc), imagecol = cc$imagecol, imagerow = cc$imagerow),
            file.path(outdir, paste0(f, "_coords.csv")), row.names = FALSE)

  # 3) 解卷积比例
  md <- obj@meta.data[, deconv_cols, drop = FALSE]
  write.csv(data.frame(barcode = rownames(md), md),
            file.path(outdir, paste0(f, "_deconv.csv")), row.names = FALSE)

  cat("  coords:", nrow(cc), "| deconv:", nrow(md), "\n")
}
cat("export done\n")
