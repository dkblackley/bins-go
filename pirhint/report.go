package pirhint

import (
	"fmt"
	"io"
	"strings"
)

// HumanBytes prints 1536 as "1.5 KB", 3.2e9 as "3.0 GB".
func HumanBytes(b float64) string {
	units := []string{"B", "KB", "MB", "GB", "TB"}
	i := 0
	for b >= 1024 && i < len(units)-1 {
		b /= 1024
		i++
	}
	if b >= 100 || i == 0 {
		return fmt.Sprintf("%.0f %s", b, units[i])
	}
	return fmt.Sprintf("%.1f %s", b, units[i])
}

// WriteMarkdown prints one row per layout: total client storage, its change
// against the first layout, the per-stage split and the server DB size.
func WriteMarkdown(w io.Writer, title string, rs []LayoutResult) {
	stages := StageNames(rs)
	fmt.Fprintf(w, "\n### %s\n\n", title)
	head := []string{"layout", "client", "vs first"}
	for _, s := range stages {
		head = append(head, s)
	}
	head = append(head, "server DB", "online/search", "refresh/search", "note")
	fmt.Fprintf(w, "| %s |\n|%s\n", strings.Join(head, " | "), strings.Repeat("---|", len(head)))
	ref := rs[0].Client
	for _, r := range rs {
		row := []string{r.Name, HumanBytes(r.Client), fmt.Sprintf("%.2fx", r.Client/ref)}
		for _, s := range stages {
			row = append(row, HumanBytes(r.Stages[s]))
		}
		row = append(row, HumanBytes(r.Server), HumanBytes(r.Online), HumanBytes(r.Offline), r.Note)
		fmt.Fprintf(w, "| %s |\n", strings.Join(row, " | "))
	}
}

// WriteDBDetail prints every DB of one layout: partitions, hint vs DB size.
func WriteDBDetail(w io.Writer, r LayoutResult) {
	fmt.Fprintf(w, "\n#### DBs of %q\n\n", r.Name)
	fmt.Fprintln(w, "| DB | rows | entry | partitions x rows | chunk / set | hints | DB | hints/DB | downloaded |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|")
	for _, d := range r.Results {
		name := d.Name
		if d.Copies > 1 {
			name = fmt.Sprintf("%s (x%d)", name, d.Copies)
		}
		fmt.Fprintf(w, "| %s | %d | %d B | %d x %d | %d / %d | %s | %s | %.2f | %v |\n",
			name, d.Rows, d.EntryBytes, d.Partitions, d.PartitionSize, d.ChunkSize, d.SetSize,
			HumanBytes(d.HintBytes), HumanBytes(d.DBBytes), d.HintOverDB(), d.Downloaded)
	}
}
