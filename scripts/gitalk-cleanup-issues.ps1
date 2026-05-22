# Close duplicate Gitalk issues (#4,#5,#6), keep #1, update its label for page.path id.
# Usage:
#   Option A: gh auth login
#   Option B: $env:GH_TOKEN = 'ghp_...'   # repo scope on blog-comments
#   .\scripts\gitalk-cleanup-issues.ps1

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Invoke-Gh {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & gh @Args
    if ($LASTEXITCODE -ne 0) {
        throw "gh failed (exit $LASTEXITCODE): gh $($Args -join ' ')"
    }
}

$repo = 'Dazzlemoon/blog-comments'
$postDir = Join-Path (Join-Path $PSScriptRoot '..') '_posts'
$postFile = Get-ChildItem -LiteralPath $postDir -Filter '2026-05-20-Koker2026-PFT*.md' | Select-Object -First 1
if (-not $postFile) { throw "Post file not found under $postDir" }
$newLabel = $postFile.BaseName
$oldLabel = 'd29b4c35d95899a97eb318c497b6dd80'

if (-not $env:GH_TOKEN) {
    $authOk = $true
    try {
        gh auth status 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { $authOk = $false }
    } catch {
        $authOk = $false
    }
    if (-not $authOk) {
        Write-Error @"
Not logged in to GitHub.

Do ONE of the following, then re-run this script:

  gh auth login

  # or (Personal Access Token with repo access):
  `$env:GH_TOKEN = 'ghp_xxxxxxxx'
  .\scripts\gitalk-cleanup-issues.ps1
"@
    }
}

Write-Host "Repo: $repo"
Write-Host "New label for issue #1: $newLabel"
Write-Host ''

Write-Host 'Closing duplicate issues #4, #5, #6 ...'
foreach ($n in 4, 5, 6) {
    Invoke-Gh issue close $n --repo $repo --comment 'Duplicate Gitalk issue (pathname md5 mismatch). Keeping #1 after migrating to stable page.path id.'
    Write-Host "  closed #$n"
}

Write-Host ''
Write-Host "Ensuring label exists: $newLabel ..."
$labelExists = gh label list --repo $repo --limit 200 --json name --jq ".[] | select(.name == `"$newLabel`") | .name" 2>$null
if (-not $labelExists) {
    Invoke-Gh label create $newLabel --repo $repo --color ededed --force
    Write-Host "  created label"
} else {
    Write-Host "  label already exists"
}

Write-Host ''
Write-Host "Updating issue #1 labels: remove $oldLabel, add $newLabel ..."
Invoke-Gh issue edit 1 --repo $repo --remove-label $oldLabel --add-label $newLabel

Write-Host ''
Write-Host 'Remaining open issues:'
Invoke-Gh issue list --repo $repo --state open
