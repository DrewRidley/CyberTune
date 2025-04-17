#!/usr/bin/env python3
"""
This script creates an enhanced dataset with real-world TTP examples from
various threat actors and reports.
"""

import json
import os
from pathlib import Path
import random
import csv

# Ensure data directory exists
Path("data/real_world").mkdir(exist_ok=True, parents=True)

# Real-world TTP examples from various threat reports and sources
REAL_WORLD_TTPS = {
    "Initial Access": [
        "APT29 used spear-phishing emails with malicious links to harvest credentials and gain initial access to targeted organizations.",
        "The Lazarus Group compromised legitimate websites frequently visited by employees (strategic web compromise) to establish initial access.",
        "Threat actors exploited CVE-2021-26855, a Microsoft Exchange Server vulnerability, to gain unauthorized access to email accounts.",
        "APT41 leveraged a zero-day vulnerability in a VPN appliance to establish persistent access to the victim network.",
        "The threat actor used valid accounts obtained through credential stuffing attacks against the organization's VPN solution.",
        "The SolarWinds supply chain attack involved trojanized updates that created a backdoor in affected organizations.",
        "FIN7 sent targeted spear-phishing emails with malicious Microsoft Office attachments to employees in specific departments.",
        "The attackers placed malicious USB drives in the company parking lot, relying on employees to connect them to workstations.",
        "Conti ransomware operators purchased access to already-compromised networks through an Initial Access Broker (IAB).",
        "The adversary exploited an insecure direct object reference in the web application to gain unauthorized access to user accounts."
    ],
    
    "Execution": [
        "Sandworm used PowerShell Empire for command and control after exploiting the initial vulnerability.",
        "APT28 leveraged Windows Management Instrumentation (WMI) to execute malicious code remotely across the network.",
        "The attackers used DLL side-loading techniques to execute malicious code through legitimate Windows processes.",
        "BlackCat ransomware operators used Group Policy to deploy and execute their payloads across the network.",
        "The threat actor used scheduled tasks to execute malware during system startup, ensuring persistence.",
        "Cozy Bear (APT29) used the Microsoft Build Engine (MSBuild) to execute malicious code, bypassing application control.",
        "APT41 executed encoded PowerShell commands to download additional payloads while evading detection systems.",
        "The attackers exploited Windows COM Hijacking to execute malicious DLLs when legitimate applications were launched.",
        "Threat actors executed malicious macros embedded in Excel spreadsheets that were opened by finance personnel.",
        "The SideWinder APT group used JavaScript to execute a series of payloads that ultimately led to a backdoor."
    ],
    
    "Persistence": [
        "APT29 modified registry run keys to maintain persistence across system reboots.",
        "The attackers installed a backdoor by creating a new Windows service configured to start automatically.",
        "Turla APT group implanted malicious code in the Windows Management Instrumentation (WMI) repository for persistence.",
        "The Lazarus Group utilized scheduled tasks to execute malware at specific time intervals or system events.",
        "APT32 achieved persistence by installing malicious browser extensions in the Chrome browser used by employees.",
        "The threat actor created a backdoored version of the corporate VPN client that was distributed to employees.",
        "Fin7 threat actors added their malware to the Windows startup folder to ensure execution at system boot.",
        "The adversary installed a bootkit that infected the master boot record, ensuring persistence across OS reinstalls.",
        "Chinese state-sponsored actors created LSASS driver backdoors to maintain access to compromised systems.",
        "The attackers leveraged Office application startup folders to execute malicious code whenever Office applications started."
    ],
    
    "Privilege Escalation": [
        "The FIN7 group used Kerberoasting attacks to extract and crack service account credentials with high privileges.",
        "APT32 exploited a time-of-check to time-of-use vulnerability in a Windows service to escalate privileges.",
        "The attackers performed UAC bypass using the Fodhelper technique to execute code with elevated privileges.",
        "Threat actors exploited the CVE-2020-1472 vulnerability (Zerologon) to obtain domain admin privileges.",
        "The adversary leveraged DLL search order hijacking to execute malicious code with elevated privileges.",
        "Sandworm exploited a buffer overflow vulnerability in a third-party driver to gain SYSTEM privileges on compromised hosts.",
        "The threat actor used a Windows named pipe impersonation technique to steal authentication tokens from privileged accounts.",
        "APT41 exploited a local privilege escalation vulnerability in the Windows Print Spooler service (PrintNightmare).",
        "Attackers used the Juicy Potato exploit to elevate privileges on Windows systems by abusing impersonation privileges.",
        "The CARBON SPIDER group used token manipulation to duplicate access tokens of high-privilege processes."
    ],
    
    "Defense Evasion": [
        "APT29 used steganography to hide malicious code within legitimate PNG image files to evade detection.",
        "The threat actor disabled Windows Defender by modifying registry keys, preventing detection of their malware.",
        "Lazarus Group operators implemented timestomping to change file timestamps and avoid forensic detection.",
        "APT41 used living-off-the-land binaries (LOLBins) like certutil.exe to download payloads and evade detection.",
        "The attackers employed binary padding techniques to modify hash values and evade signature-based detection.",
        "NOBELIUM used obfuscated JavaScript and PowerShell to hide malicious code from security analysis tools.",
        "The threat actor leveraged signed binaries to proxy execution of malicious payloads, bypassing application whitelisting.",
        "FIN7 implemented fileless malware that operated entirely in memory to avoid leaving traces on disk.",
        "The adversary deleted Windows Event Logs to cover their tracks after successful exploitation.",
        "Kimsuky APT group used legitimate cloud services for command and control, blending in with normal network traffic."
    ],
    
    "Credential Access": [
        "APT29 deployed a custom version of Mimikatz to extract credentials from LSASS memory on compromised systems.",
        "The threat actor harvested browser-saved passwords from Chrome and Firefox profiles on compromised workstations.",
        "FIN7 used keyloggers to capture credentials as users typed them, including those for financial systems.",
        "The Lazarus Group performed NTLM relay attacks to capture and replay authentication hashes across the network.",
        "Threat actors exploited the EternalBlue vulnerability to access systems and dump credential hashes.",
        "APT28 used spear-phishing emails with fake Microsoft login pages to capture Office 365 credentials.",
        "The attackers installed browser extensions that intercepted authentication traffic and stole session cookies.",
        "HAFNIUM exploited credential caches stored on Exchange servers after the initial compromise.",
        "The threat actor extracted credentials from Group Policy Preference files stored on the domain controller.",
        "Conti ransomware operators used LaZagne to recover credentials stored in various applications on compromised hosts."
    ],
    
    "Discovery": [
        "APT29 used Active Directory enumeration tools to map out the network structure and identify high-value targets.",
        "The threat actors executed PowerShell scripts to discover running processes, installed applications, and system services.",
        "BlackCat ransomware operators conducted extensive network scanning to identify vulnerable systems and critical data stores.",
        "APT41 enumerated Active Directory to identify domain administrators and service accounts for lateral movement.",
        "The adversary queried the Windows registry to discover installed security software and potential bypass methods.",
        "FIN7 operators used built-in Windows commands to discover network shares containing financial documents.",
        "Threat actors performed DNS zone transfers to map internal network resources and identify additional targets.",
        "APT32 utilized PowerShell to enumerate cloud service configuration settings and security policies.",
        "The attackers examined browser history and bookmarks to identify frequently accessed internal applications.",
        "LAPSUS$ threat actors mapped out the victim's CI/CD pipeline to identify access points to source code repositories."
    ],
    
    "Lateral Movement": [
        "APT29 used legitimate remote access tools like RDP and VNC to move laterally through the compromised environment.",
        "The threat actor exploited the ZeroLogon vulnerability to move laterally through the domain and compromise additional systems.",
        "Conti ransomware operators used SMB to spread malware to other systems across the network.",
        "The attackers used pass-the-hash techniques with stolen NTLM hashes to authenticate to remote systems.",
        "HAFNIUM threat actors exploited trust relationships between servers to move laterally to additional Exchange servers.",
        "The adversary used SSH to connect to remote Linux systems after obtaining valid credentials.",
        "FIN7 group leveraged WMI and PowerShell remoting to execute commands on remote systems for lateral movement.",
        "Threat actors exploited Kerberos delegation to move between systems while maintaining privileged access.",
        "The attackers used internal spear-phishing emails sent from compromised accounts to infect additional users.",
        "APT41 utilized legitimate systems management software to deploy malware across the network."
    ],
    
    "Collection": [
        "APT29 deployed custom tools to systematically extract emails from Microsoft Exchange servers over several months.",
        "The threat actor employed audio capture malware to record sensitive conversations in meeting rooms.",
        "LAPSUS$ exfiltrated source code repositories after compromising development environments.",
        "The attackers used PowerShell scripts to enumerate and copy sensitive files from SharePoint document libraries.",
        "FIN7 captured screenshots every 30 seconds to monitor user activity and identify valuable information.",
        "The adversary deployed clipboard monitoring malware to capture copied credentials and other sensitive data.",
        "APT41 leveraged custom malware to capture screenshots when specific banking applications were opened.",
        "Threat actors installed keyloggers specifically targeting developers with access to proprietary code.",
        "The attack group systematically extracted customer databases containing personally identifiable information.",
        "Maze ransomware operators used automated tools to identify and collect sensitive documents before encryption."
    ],
    
    "Command and Control": [
        "APT29 used HTTPS with custom encryption to communicate with command and control servers, blending with normal traffic.",
        "The threat actors implemented DNS tunneling to exfiltrate data and receive commands without raising alerts.",
        "Lazarus Group used compromised WordPress sites as proxies for their command and control infrastructure.",
        "The attackers utilized legitimate cloud services like GitHub and Dropbox for command and control communications.",
        "NOBELIUM implemented a multi-layer command and control infrastructure with several fallback mechanisms.",
        "The threat actor encrypted command and control traffic using a custom algorithm to evade network monitoring.",
        "APT41 used steganography to hide command and control instructions within image files posted to social media.",
        "The adversary implemented domain fronting techniques to disguise malicious traffic as legitimate web services.",
        "FIN7 used multiple encrypted messaging channels for different phases of their operation.",
        "The attackers utilized scheduled polling intervals with jitter to avoid detection of regular communication patterns."
    ],
    
    "Exfiltration": [
        "APT29 used encrypted ZIP archives to package stolen data before exfiltration to avoid detection.",
        "The threat actor exfiltrated data using authenticated proxies to appear as legitimate business traffic.",
        "Lazarus Group operators exfiltrated data over alternative protocols like DNS and ICMP to evade controls.",
        "The attackers scheduled data exfiltration during business hours to blend with normal network activity.",
        "FIN7 exfiltrated credit card data in small chunks over several days to avoid triggering data loss prevention tools.",
        "The adversary used steganography to hide stolen data in image files uploaded to public websites.",
        "Threat actors compromised a trusted third-party vendor to exfiltrate data through an established connection.",
        "APT41 used custom multi-stage exfiltration tools that encoded and compressed data before transmission.",
        "The attackers used legitimate cloud storage services to exfiltrate data, appearing as normal business activity.",
        "LAPSUS$ used specialized tools to compress and encrypt stolen source code before exfiltration."
    ],
    
    "Impact": [
        "Conti ransomware operators encrypted critical databases with RSA-2048 after exfiltrating sensitive information.",
        "The threat actor deployed wiper malware that corrupted the master boot record on critical systems.",
        "NotPetya malware spread rapidly through the network, rendering systems inoperable across multiple continents.",
        "The attackers modified source code in the CI/CD pipeline, introducing subtle vulnerabilities in the product.",
        "Maze ransomware operators published stolen data on their leak site after the organization refused to pay.",
        "The adversary manipulated database records to create fraudulent financial transactions.",
        "Threat actors defaced the organization's website with politically motivated messages.",
        "The attackers manipulated SCADA systems controlling industrial equipment, causing physical damage.",
        "LockBit ransomware encrypted backups before encrypting production systems to prevent recovery.",
        "The threat actor used stolen credentials to make unauthorized wire transfers to overseas accounts."
    ]
}

def save_real_world_ttps():
    """Save the real-world TTPs to JSON file"""
    output_path = "data/real_world/real_world_ttps.json"
    with open(output_path, 'w') as f:
        json.dump(REAL_WORLD_TTPS, f, indent=2)
    
    print(f"Saved {sum(len(examples) for examples in REAL_WORLD_TTPS.values())} real-world TTP examples to {output_path}")
    
    # Also create individual files for each category
    os.makedirs("data/real_world/categories", exist_ok=True)
    for category, examples in REAL_WORLD_TTPS.items():
        with open(f"data/real_world/categories/{category}.txt", 'w') as f:
            for example in examples:
                f.write(f"{example}\n")

def create_training_pairs():
    """Create training pairs from real-world TTPs for fine-tuning"""
    pairs = []
    categories = list(REAL_WORLD_TTPS.keys())
    
    # Create positive pairs (same category)
    for category, examples in REAL_WORLD_TTPS.items():
        for i, example1 in enumerate(examples):
            # Pair with other examples in same category
            for j, example2 in enumerate(examples):
                if i != j:  # Different examples, same category
                    pairs.append({
                        "sentence1": example1,
                        "sentence2": example2,
                        "label": 1.0  # Similar (same category)
                    })
    
    # Create negative pairs (different categories)
    for i, category1 in enumerate(categories):
        for j, category2 in enumerate(categories):
            if i != j:  # Different categories
                # Select random examples from each category
                examples1 = REAL_WORLD_TTPS[category1]
                examples2 = REAL_WORLD_TTPS[category2]
                
                # Create up to 5 negative pairs for each category pair
                for _ in range(min(5, len(examples1) * len(examples2))):
                    example1 = random.choice(examples1)
                    example2 = random.choice(examples2)
                    
                    pairs.append({
                        "sentence1": example1,
                        "sentence2": example2,
                        "label": 0.0  # Not similar (different categories)
                    })
    
    # Shuffle the pairs
    random.shuffle(pairs)
    
    # Save pairs
    output_path = "data/real_world/training_pairs.json"
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    print(f"Created {len(pairs)} training pairs from real-world TTPs")
    
    # Also save as CSV for easier viewing
    with open("data/real_world/training_pairs.csv", 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["sentence1", "sentence2", "label"])
        for pair in pairs:
            writer.writerow([pair["sentence1"], pair["sentence2"], pair["label"]])

def combine_datasets():
    """Combine synthetic and real-world datasets for comprehensive training"""
    # Load synthetic dataset if it exists
    synthetic_path = "data/train.json"
    real_world_path = "data/real_world/training_pairs.json"
    
    synthetic_data = []
    if os.path.exists(synthetic_path):
        with open(synthetic_path, 'r') as f:
            synthetic_data = json.load(f)
    
    # Load real-world dataset
    with open(real_world_path, 'r') as f:
        real_world_data = json.load(f)
    
    # Combine datasets
    combined_data = synthetic_data + real_world_data
    random.shuffle(combined_data)
    
    # Split into train/dev/test
    train_ratio, dev_ratio = 0.8, 0.1
    train_size = int(len(combined_data) * train_ratio)
    dev_size = int(len(combined_data) * dev_ratio)
    
    train_data = combined_data[:train_size]
    dev_data = combined_data[train_size:train_size + dev_size]
    test_data = combined_data[train_size + dev_size:]
    
    # Save combined datasets
    with open('data/combined_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('data/combined_dev.json', 'w') as f:
        json.dump(dev_data, f, indent=2)
    
    with open('data/combined_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Combined dataset created with {len(combined_data)} pairs:")
    print(f"  - Train: {len(train_data)}")
    print(f"  - Dev: {len(dev_data)}")
    print(f"  - Test: {len(test_data)}")

if __name__ == "__main__":
    # Save real-world TTPs to files
    save_real_world_ttps()
    
    # Create training pairs
    create_training_pairs()
    
    # Combine with synthetic dataset
    combine_datasets() 