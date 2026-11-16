package pianopir

import "testing"

func TestBatchPIRDownloadIfCheaper(t *testing.T) {
	defer func(d bool) { DownloadIfCheaper = d }(DownloadIfCheaper)
	DownloadIfCheaper = true

	raw := make([][]uint64, 3000)
	for i := range raw {
		raw[i] = []uint64{uint64(i), uint64(i) * 7}
	}
	raw[5] = nil // an empty bin
	pir := NewSimpleBatchPianoPIR(3000, 2, 16, 48, raw, 20, 1)
	if !pir.Downloaded() {
		t.Fatal("3000 rows over 24 partitions should be downloaded")
	}
	pir.Preprocessing()
	resp, err := pir.Query([]uint64{0, 5, 2999})
	if err != nil {
		t.Fatal(err)
	}
	if resp[0][1] != 0 || resp[2][0] != 2999 || resp[2][1] != 2999*7 || len(resp[1]) != 2 || resp[1][0] != 0 {
		t.Fatalf("wrong rows: %v", resp)
	}
	meta := pir.PrintInfo()
	if meta["ClientStorageMB"] == "" || meta["TotalByteSent"] != "0" || meta["DownloadedDB"] != "true" {
		t.Fatalf("metadata: %v", meta)
	}
	if pir.LocalStorageSize() != 3000*16 || pir.FinishedBatchNum+1 >= pir.SupportBatchNum {
		t.Fatalf("storage %v, support %v", pir.LocalStorageSize(), pir.SupportBatchNum)
	}
}
