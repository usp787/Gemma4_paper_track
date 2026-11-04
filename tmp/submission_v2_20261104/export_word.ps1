param(
    [Parameter(Mandatory=$true)][string]$DocxPath,
    [Parameter(Mandatory=$true)][string]$PdfPath
)
$ErrorActionPreference = 'Stop'
$DocxPath = [IO.Path]::GetFullPath($DocxPath)
$PdfPath = [IO.Path]::GetFullPath($PdfPath)
$expectedDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\paper\submission_v2'))
if ([IO.Path]::GetDirectoryName($DocxPath) -ne $expectedDirectory -or [IO.Path]::GetDirectoryName($PdfPath) -ne $expectedDirectory) {
    throw 'The export paths must stay in paper\submission_v2.'
}
$word = $null
$document = $null
try {
    try {
        $existing = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
        if (-not $existing.Visible) {
            foreach ($openDocument in $existing.Documents) {
                if ($openDocument.FullName -eq $DocxPath -and $openDocument.Saved) {
                    $word = $existing
                    $document = $openDocument
                    break
                }
            }
        }
        if ($null -eq $word) {
            [Runtime.InteropServices.Marshal]::ReleaseComObject($existing) | Out-Null
        }
    }
    catch [Runtime.InteropServices.COMException] {}
    if ($null -eq $word) {
    $word = New-Object -ComObject Word.Application
    }
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.UpdateLinksAtOpen = $false
    if ($null -eq $document) {
        $document = $word.Documents.Open($DocxPath, $false, $false)
    }
    $document.Repaginate()
    $document.Fields.Update() | Out-Null
    foreach ($section in $document.Sections) {
        $section.Footers.Item(1).Range.Fields.Update() | Out-Null
    }
    $document.Save()
    $document.ExportAsFixedFormat($PdfPath, 17)
    [PSCustomObject]@{Docx=$DocxPath; Pdf=$PdfPath; Pages=$document.ComputeStatistics(2); Words=$document.ComputeStatistics(0)} | ConvertTo-Json
}
finally {
    if ($null -ne $document) {
        $saveChanges = 0
        $document.Close([ref]$saveChanges)
        [Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($null -ne $word) {
        $saveChanges = 0
        $word.Quit([ref]$saveChanges)
        [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
