# BM25 MSMarco

This is a Go implementation of BM25 ranking for the MSMarco dataset, converted from the Python implementation.

## Prerequisites

1. Install Go:
   - For Ubuntu/Debian: `sudo apt-get install golang-go`
   - For macOS: `brew install go`
   - For Windows: Download installer from [golang.org](https://golang.org/dl/)

2. Verify installation:
   ```bash
   go version
   ```

## Project Structure

```
.
├── cmd/            # Command-line applications
├── internal/       # Private application and library code
│   └── bm25/      # BM25 tree implementation
├── go.mod         # Go module definition
├── run_go.sh.     # build and run script 
└── README.md      # Project documentation
```

## Getting Started

1. Clone the repository
2. Navigate to the project directory
3. Download the data folder at [Google Drive](https://drive.google.com/file/d/1hqjyfew5o6QWGynvUshe44GvPLyX1NwH/view?usp=sharing)
3. Initialize the Go module:
   ```bash
   go mod init bm25-msmarco
   ```
4. Run the application (once implemented):
   ```bash
   ./run_go
