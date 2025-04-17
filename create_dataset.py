#!/usr/bin/env python3
"""
This script creates a dataset for fine-tuning a sentence transformer model
for cybersecurity TTP (Tactics, Techniques, and Procedures) correlation.
It generates synthetic examples based on MITRE ATT&CK categories.
"""

import os
import json
import random
import csv
from pathlib import Path

# Ensure output directory exists
Path("data").mkdir(exist_ok=True)

# Example TTP categories based on MITRE ATT&CK framework
TTP_CATEGORIES = {
    "Initial Access": [
        "Phishing emails with malicious attachments targeting organization employees",
        "Exploitation of public-facing applications with known vulnerabilities",
        "Use of valid accounts obtained through credential stuffing attacks",
        "Supply chain compromise through third-party software providers",
        "Spear phishing links embedded in tailored emails to specific employees",
        "Drive-by compromise through browser vulnerabilities",
        "External remote services exploitation using brute force techniques",
        "Hardware additions like USB devices planted in physical locations",
    ],
    
    "Execution": [
        "PowerShell scripts executed to run malicious commands",
        "Command-line interface usage to run unauthorized commands",
        "Scheduled task creation to maintain persistence",
        "Windows Management Instrumentation (WMI) for remote code execution",
        "Dynamic data exchange protocol exploitation in Office documents",
        "Execution via API calls to evade detection mechanisms",
        "Use of signed binary proxy execution through legitimate tools",
        "Execution through module load to bypass application controls",
    ],
    
    "Persistence": [
        "Registry run keys modification for automatic execution at startup",
        "Creation of new services configured to execute at system boot",
        "Scheduled tasks added to maintain access after system reboots",
        "Modification of startup folder with malicious executables",
        "Account manipulation by adding privileges to compromised accounts",
        "BITS jobs created to maintain persistence and evade detection",
        "Bootkit installation to survive operating system reinstallation",
        "Browser extensions added to maintain access through web browsers",
    ],
    
    "Privilege Escalation": [
        "Exploitation of access token manipulation to elevate privileges",
        "Bypass of user account control through known vulnerabilities",
        "DLL search order hijacking to execute with elevated privileges",
        "Exploitation of setuid and setgid bits on executable files",
        "Abuse of sudo privileges through configuration weaknesses",
        "Process injection to execute code in the context of another process",
        "Scheduled task manipulation to gain system-level privileges",
        "Exploitation of vulnerable kernel drivers for privilege escalation",
    ],
    
    "Defense Evasion": [
        "Indicator removal through log clearing and file deletion",
        "Timestomping of files to hide modification times",
        "Binary padding to avoid hash-based detection mechanisms",
        "Code signing certificates used to sign malicious executables",
        "Disabling security tools through registry modifications",
        "Exploitation of rootkits to hide implants and backdoors",
        "Use of alternate data streams to hide data in NTFS file systems",
        "Port knocking techniques to hide open ports from scans",
    ],
    
    "Credential Access": [
        "Brute force attacks against authentication systems",
        "Credential dumping from memory using Mimikatz-like tools",
        "Extraction of passwords from browser storage locations",
        "Network sniffing to capture credentials transmitted in cleartext",
        "Password spraying attacks using common passwords",
        "Kerberoasting to extract service account credentials",
        "Unsecured credential storage exploitation in configuration files",
        "Forced authentication through fake SMB servers",
    ],
    
    "Discovery": [
        "Active scanning of internal networks to map infrastructure",
        "System service discovery to identify potential vulnerabilities",
        "Account enumeration to find valid user accounts",
        "Application window discovery to identify running applications",
        "Browser bookmark discovery to gather intelligence on interests",
        "File and directory discovery to locate sensitive information",
        "Query of registry to identify installed software and configurations",
        "Network share discovery to locate accessible data repositories",
    ],
    
    "Lateral Movement": [
        "Use of remote services like RDP for internal network movement",
        "Exploitation of internal spearphishing with malicious links",
        "Pass-the-hash attacks using stolen password hashes",
        "Use of remote file copy to transfer tools between systems",
        "Exploitation of shared webroot in web servers",
        "Taint shared content with malicious payloads",
        "Use of alternate authentication material like SSH keys",
        "Exploitation of Windows Admin Shares with valid credentials",
    ],
    
    "Collection": [
        "Audio capture from compromised meeting room systems",
        "Automated collection of sensitive documents",
        "Data from local systems gathered using scripts",
        "Email collection from mail servers and client applications",
        "Input capture through keylogging software",
        "Screen capture to record user activities and sensitive information",
        "Data from cloud storage extracted using API access",
        "Collection of data from removable media when connected",
    ],
    
    "Command and Control": [
        "Application layer protocol use for command channel",
        "Domain fronting through trusted domains to hide traffic",
        "Data encoding using custom algorithms to obfuscate communication",
        "Fallback channels when primary communication is disrupted",
        "Multi-stage channels using different protocols for each stage",
        "Port knocking to hide open command and control channels",
        "Use of web service APIs for command and control",
        "Traffic signaling to activate dormant implants",
    ],
    
    "Exfiltration": [
        "Data transfer through alternative protocols like DNS or ICMP",
        "Exfiltration over physical medium like USB devices",
        "Use of steganography to hide data in images or audio files",
        "Scheduled transfer of data during off-hours to avoid detection",
        "Exfiltration over command and control channel to reduce footprint",
        "Data compressed before exfiltration to reduce transfer size",
        "Encrypted data exfiltration using custom encryption schemes",
        "Traffic signaling to start data exfiltration when safe",
    ],
    
    "Impact": [
        "Data destruction through secure deletion techniques",
        "Denial of service attacks against critical infrastructure",
        "Disk wipe attacks targeting master boot records",
        "Data encryption for ransom using sophisticated ransomware",
        "Defacement of websites to demonstrate compromise",
        "Resource hijacking for cryptocurrency mining",
        "Manipulation of network traffic to disrupt operations",
        "Firmware corruption to render systems inoperable",
    ]
}

# Create similar examples with variations
def create_variations(example, num_variations=3):
    prefixes = [
        "The threat actor engaged in ", 
        "Analysts observed ", 
        "Evidence suggests ", 
        "The malware performs ", 
        "Investigation revealed ", 
        "Security logs showed ", 
        "The attack involved ", 
        "Forensic analysis identified "
    ]
    
    suffixes = [
        " according to incident responders",
        " as part of the attack chain",
        " to achieve their objectives",
        " which was previously unreported",
        " similar to known APT behaviors",
        " indicating sophisticated actors",
        " bypassing standard security measures",
        " during the compromise"
    ]
    
    variations = []
    for _ in range(num_variations):
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes) if random.random() > 0.5 else ""
        variation = f"{prefix}{example.lower()}{suffix}"
        variations.append(variation)
    
    return variations

# Create dataset
def create_dataset():
    # Create pairs dataset (positive pairs)
    pairs = []
    
    for category, examples in TTP_CATEGORIES.items():
        for i, example in enumerate(examples):
            # Create variations of this example
            variations = create_variations(example)
            
            # Create positive pairs (same category)
            for variation in variations:
                # Pair with original
                pairs.append({
                    "sentence1": example,
                    "sentence2": variation,
                    "label": 1.0  # Similar (same category)
                })
                
                # Pair with other examples in same category
                for j, other_example in enumerate(examples):
                    if i != j:
                        pairs.append({
                            "sentence1": variation,
                            "sentence2": other_example,
                            "label": 1.0  # Similar (same category)
                        })
    
    # Add negative pairs (different categories)
    categories = list(TTP_CATEGORIES.keys())
    for i, category1 in enumerate(categories):
        for j, category2 in enumerate(categories):
            if i != j:  # Different categories
                # Select random examples from each category
                examples1 = TTP_CATEGORIES[category1]
                examples2 = TTP_CATEGORIES[category2]
                
                # Create negative pairs
                for _ in range(min(10, len(examples1) * len(examples2))):
                    example1 = random.choice(examples1)
                    example2 = random.choice(examples2)
                    
                    pairs.append({
                        "sentence1": example1,
                        "sentence2": example2,
                        "label": 0.0  # Not similar (different categories)
                    })
    
    # Shuffle the pairs
    random.shuffle(pairs)
    
    # Split into train/dev/test
    train_ratio, dev_ratio = 0.8, 0.1
    train_size = int(len(pairs) * train_ratio)
    dev_size = int(len(pairs) * dev_ratio)
    
    train_pairs = pairs[:train_size]
    dev_pairs = pairs[train_size:train_size + dev_size]
    test_pairs = pairs[train_size + dev_size:]
    
    # Save datasets
    with open('data/train.json', 'w') as f:
        json.dump(train_pairs, f, indent=2)
    
    with open('data/dev.json', 'w') as f:
        json.dump(dev_pairs, f, indent=2)
    
    with open('data/test.json', 'w') as f:
        json.dump(test_pairs, f, indent=2)
    
    # Also create a CSV version for easier viewing
    with open('data/train.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence1', 'sentence2', 'label'])
        for pair in train_pairs:
            writer.writerow([pair['sentence1'], pair['sentence2'], pair['label']])
    
    print(f"Created dataset with {len(pairs)} pairs:")
    print(f"  - Train: {len(train_pairs)}")
    print(f"  - Dev: {len(dev_pairs)}")
    print(f"  - Test: {len(test_pairs)}")
    
    # Create a file with categories and examples for reference
    with open('data/categories.json', 'w') as f:
        json.dump(TTP_CATEGORIES, f, indent=2)
    
    # Create raw text file for each category (useful for some training methods)
    os.makedirs('data/categories', exist_ok=True)
    for category, examples in TTP_CATEGORIES.items():
        with open(f'data/categories/{category}.txt', 'w') as f:
            for example in examples:
                f.write(f"{example}\n")
                for variation in create_variations(example):
                    f.write(f"{variation}\n")

if __name__ == "__main__":
    create_dataset()
    print("Dataset creation completed.") 