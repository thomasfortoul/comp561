# Install Bioconductor packages
if (!requireNamespace("BiocManager", quietly=TRUE))
    install.packages("BiocManager")

# Load libraries
library(DNAshapeR)
library(BSgenome.Hsapiens.UCSC.hg38)  # Reference genome
library(data.table)  # Efficient file reading and manipulation

# Load the human genome
genome <- BSgenome.Hsapiens.UCSC.hg38

# Function to validate genomic coordinates
validate_coordinates <- function(regions, genome) {
  # Get chromosome sizes from the genome
  chr_lengths <- seqlengths(genome)
  
  # Ensure coordinates are within bounds
  valid_regions <- regions[
    chr %in% names(chr_lengths) & 
    start > 0 & end > 0 & 
    end <= chr_lengths[chr]
  ]
  
  return(valid_regions)
}

# Load and process the first file
file1 <- fread("Project/CTCF_filtered.txt", header = FALSE, col.names = c("id", "chr", "start", "end", "feature", "score", "strand"))
file1 <- file1[sample(.N, min(50000, .N))]  # Sample 10,000 rows
file1 <- validate_coordinates(file1, genome)  # Validate coordinates

# # Load and process the second file
file2 <- fread("Project/negative_matches_split.txt", header = FALSE, col.names = c("chr", "start", "end"))
file2 <- file2[sample(.N, min(50000, .N))]  # Sample 10,000 rows
file2 <- validate_coordinates(file2, genome)  # Validate coordinates
# Strand-aware sequence extraction for file1
extract_sequences_with_strand <- function(regions, genome) {
  sequences <- lapply(1:nrow(regions), function(i) {
    seq <- getSeq(genome, regions$chr[i], regions$start[i], regions$end[i])
    if (regions$strand[i] == "-") {
      seq <- reverseComplement(seq)
    }
    return(seq)
  })
  names(sequences) <- paste0(regions$chr, ":", regions$start, "-", regions$end,regions$strand)
  return(DNAStringSet(sequences))
}

# Simple sequence extraction for file2 (no strand info)
extract_sequences <- function(regions, genome) {
  sequences <- lapply(1:nrow(regions), function(i) {
   return(getSeq(genome, regions$chr[i], regions$start[i], regions$end[i]))
  })
  names(sequences) <- paste0(regions$chr, ":", regions$start, "-", regions$end, "_shuf")
  return(DNAStringSet(sequences))
}

# Extract sequences from file1 with strand consideration
# sequences_file1 <- extract_sequences_with_strand(file1, genome)

# Extract sequences from file2 assuming + strand
sequences_file2 <- extract_sequences(file2, genome)

# Save sequences to separate FASTA files
# writeXStringSet(sequences_file1, filepath = "matches.fasta")
writeXStringSet(sequences_file2, filepath = "non_matches.fasta")

# Analyze DNA shapes for each file
shapes_file1 <- getShape("randomized_output.fasta")
shapes_file2 <- getShape("filtered_non_matches.fasta")

# Save the results
save(shapes_file1, file = "matches_DNAshapes.RData")
save(shapes_file2, file = "matches_DNAshapes.RData")

# Optional: Print summary statistics
print(summary(shapes_file1))
print(summary(shapes_file2))
