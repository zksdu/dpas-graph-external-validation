# 检查 Seurat 对象是否含 image slot / 空间坐标
suppressMessages(library(Seurat))
out <- "D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/rds_structure.txt"
con <- file(out, open = "wt")
sink(con, type = "output")
for (f in c("sectionA", "sectionB")) {
  p <- sprintf("D:/workbuddy/0911/pe-virtual-protein/data/pe_placenta/%s_visium_spotlight.rds", f)
  obj <- readRDS(p)
  cat("=====", f, "=====\n")
  cat("class:", class(obj), "\n")
  cat("slot names:", paste(slotNames(obj), collapse = ", "), "\n")
  cat("assays:", paste(names(obj@assays), collapse = ", "), "\n")
  cat("reductions:", paste(names(obj@reductions), collapse = ", "), "\n")
  cat("meta.data cols:", paste(colnames(obj@meta.data), collapse = " | "), "\n")
  if ("images" %in% slotNames(obj)) {
    cat("images:", paste(names(obj@images), collapse = ", "), "\n")
    for (nm in names(obj@images)) {
      img <- obj@images[[nm]]
      cat("--- image:", nm, "class:", class(img), "\n")
      cc <- tryCatch(GetTissueCoordinates(obj, image = nm), error = function(e) NULL)
      if (!is.null(cc)) {
        cat("    coords dim:", dim(cc), "\n")
        print(head(cc, 3))
      }
    }
  } else {
    cat("NO images slot\n")
  }
  # 检查 meta.data 里是否有坐标类列
  coord_cols <- grep("row|col|x|y|px|coord|spatial|imagerow|imagecol", colnames(obj@meta.data),
                     ignore.case = TRUE, value = TRUE)
  cat("坐标类 meta 列:", paste(coord_cols, collapse = " | "), "\n")
  if (length(coord_cols) > 0) print(head(obj@meta.data[, coord_cols, drop = FALSE], 3))
}
sink()
close(con)
cat("done\n")
