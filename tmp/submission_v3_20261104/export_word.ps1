param(
    [Parameter(Mandatory=$true)][string]$DocxPath,
    [Parameter(Mandatory=$true)][string]$PdfPath
)
$ErrorActionPreference = 'Stop'
$DocxPath = [IO.Path]::GetFullPath($DocxPath)
$PdfPath = [IO.Path]::GetFullPath($PdfPath)
$expectedDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\paper\submission_v3'))
if ([IO.Path]::GetDirectoryName($DocxPath) -ne $expectedDirectory -or [IO.Path]::GetDirectoryName($PdfPath) -ne $expectedDirectory) {
    throw 'The export paths must stay in paper\submission_v3.'
}
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.UpdateLinksAtOpen = $false
    $document = $word.Documents.Open($DocxPath, $false, $false)
    $document.Repaginate()
    $document.Fields.Update() | Out-Null
    foreach ($section in $document.Sections) {
        $section.Footers.Item(1).Range.Fields.Update() | Out-Null
    }
    $referencesStart = $null
    $introStart = $null
    $mainEnd = $null
    foreach ($paragraph in $document.Paragraphs) {
        $text = $paragraph.Range.Text.Trim()
        if ($text -eq 'References') { $referencesStart = $paragraph.Range.Start }
        if ($text -eq '1 Introduction') { $introStart = $paragraph.Range.Start }
        if ($text -eq 'GitHub Repository') { $mainEnd = $paragraph.Range.Start }
    }
    if ($null -eq $referencesStart -or $null -eq $introStart -or $null -eq $mainEnd) {
        throw 'Could not locate the word-count boundaries.'
    }
    $countedWords = $document.Range(0, $referencesStart).ComputeStatistics(0)
    $coreBodyWords = $document.Range($introStart, $mainEnd).ComputeStatistics(0)
    $document.Save()
    $document.ExportAsFixedFormat($PdfPath, 17)
    $report = [PSCustomObject]@{
        Docx=$DocxPath; Pdf=$PdfPath; Pages=$document.ComputeStatistics(2)
        TotalDocumentWords=$document.ComputeStatistics(0)
        CountedWordsBeforeReferences=$countedWords
        CoreBodyWordsSections1To9=$coreBodyWords
        Limit=3000; WordsOverLimit=[Math]::Max(0, $countedWords - 3000)
        CountScope='Title, subtitle, author line, abstract, headings, sections 1-9, captions, table, repository link, and AI disclosure. Figure artwork, references, and page numbers excluded.'
        Counter='Microsoft Word Range.ComputeStatistics(wdStatisticWords)'
    }
    $report | ConvertTo-Json | Set-Content (Join-Path $PSScriptRoot 'word_count.json') -Encoding UTF8
    $report | ConvertTo-Json
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
