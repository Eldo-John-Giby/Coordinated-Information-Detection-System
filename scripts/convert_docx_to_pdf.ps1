param (
    [string]$DocxPath = "manuscript.docx",
    [string]$PdfPath = "manuscript.pdf"
)

$absDocx = (Resolve-Path $DocxPath).Path
$absPdf = [System.IO.Path]::GetFullPath($PdfPath)

Write-Host "Converting $absDocx to $absPdf using Microsoft Word..."

$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($absDocx)
    # 17 = wdFormatPDF
    $doc.SaveAs([ref]$absPdf, [ref]17)
    $doc.Close()
    Write-Host "Successfully generated: $absPdf (Size: $((Get-Item $absPdf).Length) bytes)"
}
catch {
    Write-Error "Conversion failed: $_"
}
finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
