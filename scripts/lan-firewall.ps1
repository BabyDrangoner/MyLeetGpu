[CmdletBinding()]
param(
    [ValidateSet("Enable", "Disable", "Status")]
    [string]$Action = "Status",
    [ValidateRange(1, 65535)]
    [int]$Port = 3000,
    [string]$ListenAddress,
    [string[]]$RemoteAddress = @("LocalSubnet")
)

$ErrorActionPreference = "Stop"
$creatorId = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
$hyperVRuleName = "MyLeetGpu-LAN-$Port-HyperV"
$windowsRuleName = "MyLeetGpu-LAN-$Port-Windows"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Remove-MyLeetGpuRules {
    Get-NetFirewallHyperVRule -Name $hyperVRuleName -ErrorAction SilentlyContinue |
        Remove-NetFirewallHyperVRule -ErrorAction Stop
    Get-NetFirewallRule -Name $windowsRuleName -ErrorAction SilentlyContinue |
        Remove-NetFirewallRule -ErrorAction Stop
}

if ($Action -in @("Enable", "Disable") -and -not (Test-Administrator)) {
    throw "请在管理员 PowerShell 中运行此脚本。"
}

switch ($Action) {
    "Enable" {
        if (-not $ListenAddress) {
            throw "Enable 需要 -ListenAddress，且必须是明确的局域网 IPv4 地址。"
        }
        $parsedAddress = $null
        if (-not [Net.IPAddress]::TryParse($ListenAddress, [ref]$parsedAddress) -or
            $parsedAddress.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or
            $ListenAddress -eq "0.0.0.0" -or $ListenAddress.StartsWith("127.")) {
            throw "ListenAddress 必须是明确的非回环 IPv4 地址，不能使用 0.0.0.0。"
        }

        Remove-MyLeetGpuRules
        New-NetFirewallHyperVRule `
            -Name $hyperVRuleName `
            -DisplayName "MyLeetGpu LAN $Port (WSL Hyper-V)" `
            -Direction Inbound `
            -VMCreatorId $creatorId `
            -Protocol TCP `
            -LocalAddresses $ListenAddress `
            -LocalPorts $Port `
            -RemoteAddresses $RemoteAddress `
            -Action Allow `
            -Enabled True `
            -Profiles Any | Out-Null
        New-NetFirewallRule `
            -Name $windowsRuleName `
            -DisplayName "MyLeetGpu LAN $Port (Windows)" `
            -Direction Inbound `
            -Protocol TCP `
            -LocalAddress $ListenAddress `
            -LocalPort $Port `
            -RemoteAddress $RemoteAddress `
            -Action Allow `
            -Enabled True `
            -Profile Any | Out-Null
        Write-Host "已允许 $($RemoteAddress -join ',') 访问 http://${ListenAddress}:$Port"
    }
    "Disable" {
        Remove-MyLeetGpuRules
        Write-Host "已删除 MyLeetGpu LAN $Port 防火墙规则。"
    }
    "Status" {
        Write-Host "Hyper-V rule:"
        Get-NetFirewallHyperVRule -Name $hyperVRuleName -ErrorAction SilentlyContinue |
            Format-List Name, DisplayName, Enabled, Direction, Action, Profiles, Protocol, LocalAddresses, LocalPorts, RemoteAddresses
        Write-Host "Windows rule:"
        Get-NetFirewallRule -Name $windowsRuleName -ErrorAction SilentlyContinue |
            Format-List Name, DisplayName, Enabled, Direction, Action, Profile
    }
}
