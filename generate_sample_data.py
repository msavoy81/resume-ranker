#!/usr/bin/env python3
"""
Generates 75 realistic fake candidates for testing the resume ranker.

Distribution:
  • 30 Tier A  (~40%) — 3+ years at most recent company
  • 15 Tier B  (~20%) — <3 years, no recent degree (≤ 2022)
  • 30 Tier C  (~40%) — <3 years + recent degree (2023–2025)

Location:
  • ~40% NY metro  (NYC / Brooklyn / Queens / Hoboken / Jersey City / LIC)
  • ~60% non-local  (various US cities)

JD-fit profiles (controls keyword depth in generated PDFs):
  • deep_tier1   — LangGraph & AWS in job title line + Amazon Bedrock in bullets
                   → title line has year → depth=10 for LangGraph/AWS, depth=5 for Bedrock
                   → JD composite ~8–9
  • medium_tier1 — T1 keywords only in skills section → depth=2 → composite ~2
  • tier2_deep   — MLOps/LLMOps/CI-CD in bullets → T2 depth=5 → composite ~1.5
  • tier2_skills — DevOps/CI-CD only in skills → depth=2 → composite ~0.6
  • general_ml   — Machine Learning/LLM keywords in bullets → T3 depth=5 → composite ~0.5
  • weak         — No relevant keywords → composite ~0

Output:
  • sample_candidates.csv   — Greenhouse-style CSV
  • sample_resumes/         — one PDF per candidate
  • sample_resumes.zip      — zipped bundle ready for the ranker
"""

import zipfile
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, HRFlowable
from reportlab.lib.enums import TA_CENTER

BASE = Path(__file__).parent
RESUMES_DIR = BASE / "sample_resumes"
RESUMES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Bullet templates — keyed by fit profile, list of interchangeable sets
# ─────────────────────────────────────────────────────────────────────────────

BULLET_SETS = {
    # LangGraph + AWS in TITLE → depth=10; Amazon Bedrock in bullets → depth=5
    "deep_tier1": [
        [
            "Designed and operates LangGraph-based multi-agent workflows serving {n} production AI applications.",
            "Built CI/CD pipelines for Amazon Bedrock model deployment: prompt versioning, evaluation gates, automated rollback.",
            "Implements AWS infrastructure (IAM, Secrets Manager, CloudWatch, Lambda) for secure LLM operations and agent sandboxing.",
            "Manages evaluation pipelines measuring hallucination rates, tool-call success, and latency p99 across all LLM systems.",
            "Led Amazon Bedrock API integrations for summarization, personalization, and AI-assisted workflows; owns SLO design.",
        ],
        [
            "Architects LangGraph multi-agent orchestration layer powering {n} customer-facing AI products.",
            "Owns end-to-end LLM deployment on Amazon Bedrock: model selection, prompt versioning, canary rollouts, rollback automation.",
            "Builds CloudWatch observability dashboards tracking cost-per-request, error rates, and agent workflow completion.",
            "Implements AWS Secrets Manager and IAM least-privilege policies for all LLM service accounts and agent environments.",
            "Established evaluation framework benchmarking hallucination rates and coherence scores for every model release.",
        ],
        [
            "Leads LLMOps engineering for {n} production AI agents built on Amazon Bedrock platform.",
            "Designs CI/CD automation for prompt and model version promotion across dev/staging/prod using AWS CodePipeline.",
            "Implements sandboxed agent execution with IAM restrictions, CloudWatch alerting, and automated SLO enforcement.",
            "Owns hallucination measurement and regression testing pipelines triggered pre-deployment for every LLM release.",
            "Partners with applied science teams to operationalize new Amazon Bedrock model families with zero-downtime migrations.",
        ],
    ],
    # No T1 keywords in bullets — only in skills section → depth=2
    "medium_tier1": [
        [
            "Manages cloud ML infrastructure for {n} production model endpoints; partners with data scientists on deployment workflows.",
            "Builds deployment automation and model versioning tooling for the ML platform team.",
            "Implements CloudWatch monitoring and alerting for production AI system reliability and SLOs.",
            "Manages AWS infrastructure (SageMaker, Lambda, ECS) for model serving and experiment tracking.",
            "Supports incident response for production AI system outages; maintains operational playbooks.",
        ],
        [
            "Leads infrastructure projects for AI/ML workloads on AWS; owns platform reliability and SLO compliance.",
            "Partners with applied scientists on model deployment, performance monitoring, and A/B testing rollouts.",
            "Builds deployment pipelines and automated testing for ML model releases.",
            "Manages container orchestration (Docker, ECS) and observability stack for production ML services.",
            "Collaborates on evaluation frameworks and automated regression testing for model releases.",
        ],
    ],
    # MLOps / LLMOps / CI-CD in bullets → T2 depth=5
    "tier2_deep": [
        [
            "Manages MLOps infrastructure supporting {n}+ production ML models across dev, staging, and production environments.",
            "Builds and maintains CI/CD pipelines for model deployment with automated evaluation gates and canary releases.",
            "Implements LLMOps and DevOps best practices: GitOps, IaC with Terraform, blue-green deployments.",
            "Owns observability stack (Grafana, CloudWatch) covering latency, error rate, and cost-per-request SLOs.",
            "Leads incident response for AI system outages; runbooks reduced mean time to recovery by 35%.",
        ],
        [
            "Designs and operates MLOps platform serving {n} ML models at 99.9%+ uptime SLO.",
            "Builds CI/CD automation for model and prompt deployments; enforces evaluation gates before production promotion.",
            "Implements DevOps tooling: feature flags, automated rollback, and infrastructure-as-code with Terraform.",
            "Manages Kubernetes-based ML serving infrastructure; monitors performance with Datadog and CloudWatch.",
            "Conducts incident retrospectives and improves on-call playbooks; reduced P0 incidents 40% year-over-year.",
        ],
    ],
    # DevOps / CI-CD only in skills section → depth=2 for T2
    "tier2_skills": [
        [
            "Maintains deployment pipelines for software releases across {n}+ services using GitHub Actions and Jenkins.",
            "Manages AWS infrastructure (EC2, ECS, RDS, S3, CloudWatch) for SaaS products in production.",
            "Implements Terraform IaC and Kubernetes orchestration; achieved 99.95% uptime SLO.",
            "Built monitoring and alerting dashboards; trained engineers on observability practices.",
            "Reduced deployment cycle time by 60% through pipeline automation and standardized tooling.",
        ],
        [
            "Builds and maintains infrastructure for engineering teams; focuses on reliability and automation.",
            "Manages cloud workloads on AWS for data and application systems.",
            "Implements monitoring and logging solutions for production services.",
            "Supports container orchestration (Docker, Kubernetes) for production workloads.",
            "Automates infrastructure provisioning using Terraform and GitHub Actions.",
        ],
    ],
    # Machine Learning / LLM in bullets → T3 depth=5
    "general_ml": [
        [
            "Builds and deploys machine learning models for classification and prediction tasks using Python.",
            "Works with large language models (LLMs) for natural language processing features and internal AI tooling.",
            "Develops artificial intelligence prototypes and collaborates on productionization with platform engineers.",
            "Maintains Python-based ML pipelines and model serving endpoints on AWS.",
            "Collaborates with data teams on feature engineering, model validation, and monitoring.",
        ],
        [
            "Develops machine learning solutions for business problems using scikit-learn, PyTorch, and Python.",
            "Integrates LLM APIs for internal tooling and customer-facing artificial intelligence features.",
            "Builds AI-powered analytics and supports A/B testing for model releases.",
            "Manages experiment tracking and model versioning using MLflow.",
            "Supports model monitoring and drift detection pipelines in production.",
        ],
    ],
    # No relevant keywords
    "weak": [
        [
            "Develops backend services and APIs serving internal and external customers.",
            "Collaborates with product and data teams on technical requirements and system design.",
            "Maintains cloud-hosted services and deployment workflows on AWS.",
            "Participates in code review, pair programming, and sprint planning ceremonies.",
            "Supports production systems and responds to operational incidents.",
        ],
        [
            "Builds Python REST APIs for platform features and data integrations.",
            "Uses AWS services (S3, Lambda, EC2) for application hosting and basic data storage.",
            "Supports continuous integration processes using GitHub Actions and Jenkins.",
            "Maintains internal tooling and supports engineering team productivity.",
            "Actively learning ML concepts; completed AWS Cloud Practitioner certification.",
        ],
    ],
}

PREV_BULLETS = {
    "deep_tier1": [
        "Built production ML infrastructure on AWS supporting recommendation and personalization systems.",
        "Implemented Kubernetes-based model serving platform handling 10K+ requests per minute.",
        "Migrated legacy data pipelines to AWS, improving reliability and reducing latency by 40%.",
    ],
    "medium_tier1": [
        "Designed cloud infrastructure for data processing and ML workloads on AWS.",
        "Supported model deployment pipelines and experiment tracking using MLflow.",
        "Built monitoring dashboards and alerting for production ML systems.",
    ],
    "tier2_deep": [
        "Built automated deployment pipelines reducing release cycle from days to under 2 hours.",
        "Managed Kubernetes clusters and container orchestration for microservices platform.",
        "Implemented IaC using Terraform; reduced infrastructure provisioning time by 70%.",
    ],
    "tier2_skills": [
        "Automated deployment workflows and maintained CI/CD tooling for engineering teams.",
        "Supported cloud infrastructure and developer tooling including Docker and GitHub Actions.",
        "Managed AWS infrastructure for growing engineering team.",
    ],
    "general_ml": [
        "Built data pipelines and analytics dashboards for product and marketing stakeholders.",
        "Developed ML models for classification and recommendation use cases using scikit-learn.",
        "Collaborated with data teams on feature engineering and model evaluation.",
    ],
    "weak": [
        "Developed REST APIs and backend services for consumer-facing web applications.",
        "Supported engineering team with bug fixes, code reviews, and feature development.",
        "Built data integrations and internal tooling using Python and SQL.",
    ],
}

INTERNSHIP_BULLETS = [
    "Built features for the platform team during a 12-week internship; shipped 3 production improvements.",
    "Contributed to backend services and gained hands-on experience with Python, AWS, and agile development.",
    "Participated in code reviews, sprint planning, and engineering design discussions.",
]

SKILLS_MAP = {
    "deep_tier1": (
        "AWS (Bedrock, SageMaker, CloudWatch, Lambda, ECS, Secrets Manager, IAM, CodePipeline), "
        "LangGraph, LangChain, Python, Docker, Kubernetes, Terraform, GitHub Actions, MLflow, "
        "CI/CD, observability, SLO design, evaluation frameworks, infrastructure-as-code, prompt engineering"
    ),
    "medium_tier1": (
        "AWS (Bedrock, SageMaker, EC2, S3, Lambda, CloudWatch), LangGraph, LangChain, "
        "Python, Docker, Terraform, MLflow, monitoring, GitHub Actions"
    ),
    "tier2_deep": (
        "MLOps, LLMOps, DevOps, CI/CD, AWS (EC2, ECS, Lambda, S3, CloudWatch, CodePipeline), "
        "Python, Terraform, Kubernetes, Docker, GitHub Actions, Grafana, Datadog, "
        "observability, SLO design, incident response, infrastructure-as-code"
    ),
    "tier2_skills": (
        "DevOps, CI/CD, AWS (EC2, S3, ECS, Lambda, CloudWatch, RDS), Python, Docker, "
        "Terraform, Kubernetes, GitHub Actions, Jenkins, Bash, monitoring, alerting"
    ),
    "general_ml": (
        "Python, Machine Learning, LLMs, scikit-learn, TensorFlow, PyTorch, "
        "AWS (EC2, S3, SageMaker — basic), SQL, pandas, NumPy, Jupyter, MLflow, data analysis"
    ),
    "weak": (
        "Python, JavaScript, REST APIs, SQL, basic AWS (EC2, S3, Lambda), Git, "
        "Docker (basic), Bash, no ML/LLM production experience"
    ),
}

SUMMARY_TEMPLATES = {
    "deep_tier1": (
        "{title} with {years}+ years building and operating LLM infrastructure on AWS. "
        "Deep expertise with Amazon Bedrock, LangGraph, and CI/CD pipelines for production AI systems. "
        "Proven record delivering agentic AI platforms at scale with robust observability and evaluation frameworks."
    ),
    "medium_tier1": (
        "{title} with {years} years of experience in cloud infrastructure and ML systems. "
        "Hands-on with AWS services including Amazon Bedrock and LangGraph for agentic AI workloads. "
        "Strong background in platform reliability, MLOps tooling, and production deployment."
    ),
    "tier2_deep": (
        "{title} with {years} years specializing in MLOps, LLMOps, and DevOps for AI/ML workloads. "
        "Deep expertise in CI/CD pipelines, Kubernetes orchestration, and AWS infrastructure. "
        "Track record of improving production reliability and observability for ML systems."
    ),
    "tier2_skills": (
        "{title} with {years} years of experience in DevOps, CI/CD, and cloud infrastructure engineering. "
        "Strong AWS and containerization background; growing focus on AI/ML platform operations. "
        "Committed to reliability, automation, and operational excellence."
    ),
    "general_ml": (
        "{title} with {years} years of experience building machine learning models and AI-powered features. "
        "Experienced with large language models (LLMs), Python ML frameworks, and cloud deployment on AWS. "
        "Passionate about applied AI and interested in production ML infrastructure and MLOps."
    ),
    "weak": (
        "{title} with {years} years of software development and cloud infrastructure experience. "
        "Strong Python and API development background; actively learning cloud-native and AI engineering concepts. "
        "Interested in expanding into ML/LLM infrastructure and DevOps engineering."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Candidate specs — 75 candidates
# ─────────────────────────────────────────────────────────────────────────────
# Keys: name, email, location (Greenhouse CSV), stage, applied,
#       title (current), company (current), cloc (company city),
#       rstart (role start year str), rend (role end, default "Present"),
#       prev_title, prev_company, ploc, pyears,
#       is_internship (bool, optional),
#       edu (education line), fit (profile key)

SPECS = [
    # ═══════════════════════════════════════════════════════════════════════
    # TIER A — most recent role 3+ years  (rstart 2019–2022)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Tier A · NY Metro · deep_tier1 ──────────────────────────────────

    dict(name="Amara Osei",
         email="a.osei@proton.me", location="Brooklyn, NY",
         stage="Second Interview", applied="2025-04-02",
         title="Senior LLMOps & LangGraph Engineer", company="Condé Nast",
         cloc="New York, NY", rstart="2021", rend="Present",
         prev_title="ML Platform Engineer", prev_company="NBCUniversal",
         ploc="New York, NY", pyears="2018 – 2021",
         edu="B.S. Computer Science, Cornell University, 2018",
         fit="deep_tier1"),

    dict(name="David Kim",
         email="david.kim@icloud.com", location="Manhattan, NY",
         stage="First Interview", applied="2025-04-03",
         title="AI Infrastructure Lead — LangGraph & AWS", company="Goldman Sachs",
         cloc="New York, NY", rstart="2020", rend="Present",
         prev_title="ML Engineer", prev_company="Two Sigma",
         ploc="New York, NY", pyears="2018 – 2020",
         edu="M.S. Computer Science, Columbia University, 2018; B.S. CS, UIUC, 2017",
         fit="deep_tier1"),

    dict(name="Sofia Ramirez",
         email="sofia.ramirez@gmail.com", location="Hoboken, NJ",
         stage="Second Interview", applied="2025-04-04",
         title="LLMOps & LangGraph Platform Engineer", company="Spotify",
         cloc="New York, NY", rstart="2022", rend="Present",
         prev_title="Software Engineer", prev_company="SiriusXM",
         ploc="Hoboken, NJ", pyears="2020 – 2022",
         edu="B.S. Computer Science, Rutgers University, 2020",
         fit="deep_tier1"),

    # ── Tier A · NY Metro · medium_tier1 ────────────────────────────────

    dict(name="Isabella Santos",
         email="i.santos@outlook.com", location="New York, NY",
         stage="Phone Screen", applied="2025-04-05",
         title="ML Platform Engineer", company="Bloomberg L.P.",
         cloc="New York, NY", rstart="2022", rend="Present",
         prev_title="Data Engineer", prev_company="Fiserv",
         ploc="Jersey City, NJ", pyears="2020 – 2022",
         edu="B.S. Software Engineering, Stony Brook University, 2020",
         fit="medium_tier1"),

    dict(name="Michael Torres",
         email="mj.torres@gmail.com", location="Jersey City, NJ",
         stage="Applied", applied="2025-04-07",
         title="Senior ML Engineer", company="John Wiley & Sons",
         cloc="Hoboken, NJ", rstart="2021", rend="Present",
         prev_title="Backend Engineer", prev_company="WebMD Health",
         ploc="New York, NY", pyears="2019 – 2021",
         edu="B.S. Computer Science, NJIT, 2019",
         fit="medium_tier1"),

    # ── Tier A · NY Metro · tier2_deep ──────────────────────────────────

    dict(name="James Okafor",
         email="j.okafor@proton.me", location="New York, NY",
         stage="Applied", applied="2025-04-06",
         title="MLOps Lead", company="Pfizer",
         cloc="New York, NY", rstart="2020", rend="Present",
         prev_title="DevOps Engineer", prev_company="Merck",
         ploc="Kenilworth, NJ", pyears="2017 – 2020",
         edu="B.S. Computer Engineering, NYU Tandon, 2017",
         fit="tier2_deep"),

    dict(name="Yuki Tanaka",
         email="yuki.tanaka@gmail.com", location="Long Island City, NY",
         stage="Applied", applied="2025-04-08",
         title="AI Platform Engineer", company="NFL",
         cloc="New York, NY", rstart="2022", rend="Present",
         prev_title="MLOps Engineer", prev_company="The Athletic",
         ploc="New York, NY", pyears="2020 – 2022",
         edu="B.S. Computer Science, CUNY City College, 2020",
         fit="tier2_deep"),

    # ── Tier A · NY Metro · tier2_skills ────────────────────────────────

    dict(name="Rachel Goldman",
         email="r.goldman@icloud.com", location="New York, NY",
         stage="Applied", applied="2025-04-09",
         title="DevOps Lead", company="Macy's Technology",
         cloc="New York, NY", rstart="2021", rend="Present",
         prev_title="DevOps Engineer", prev_company="Saks Fifth Avenue",
         ploc="New York, NY", pyears="2019 – 2021",
         edu="B.S. Information Systems, NYU Stern, 2019",
         fit="tier2_skills"),

    # ── Tier A · NY Metro · general_ml ──────────────────────────────────

    dict(name="Carlos Mendes",
         email="carlos.mendes@gmail.com", location="Queens, NY",
         stage="Applied", applied="2025-04-10",
         title="ML Research Scientist", company="Cornell Tech",
         cloc="New York, NY", rstart="2022", rend="Present",
         prev_title="Research Engineer", prev_company="NYU Courant",
         ploc="New York, NY", pyears="2020 – 2022",
         edu="Ph.D. Computer Science, NYU, 2022; B.S. Mathematics, Queens College, 2017",
         fit="general_ml"),

    # ── Tier A · NY Metro · weak ─────────────────────────────────────────

    dict(name="Priya Nair",
         email="priya.nair@icloud.com", location="Hoboken, NJ",
         stage="Applied", applied="2025-04-11",
         title="Principal Engineer", company="Morgan Stanley",
         cloc="New York, NY", rstart="2019", rend="Present",
         prev_title="Software Engineer III", prev_company="Barclays",
         ploc="New York, NY", pyears="2015 – 2019",
         edu="B.S. Computer Science, University of Pennsylvania, 2015",
         fit="weak"),

    dict(name="Thomas Bennett",
         email="t.bennett@gmail.com", location="Jersey City, NJ",
         stage="Applied", applied="2025-04-12",
         title="Backend Lead", company="IAC",
         cloc="New York, NY", rstart="2020", rend="Present",
         prev_title="Senior Engineer", prev_company="Rent the Runway",
         ploc="Brooklyn, NY", pyears="2017 – 2020",
         edu="B.S. Software Engineering, Stevens Institute of Technology, 2017",
         fit="weak"),

    dict(name="Keisha Brown",
         email="keisha.brown@proton.me", location="New York, NY",
         stage="Applied", applied="2025-04-13",
         title="Data Engineer", company="Collibra",
         cloc="New York, NY", rstart="2021", rend="Present",
         prev_title="ETL Engineer", prev_company="FTI Consulting",
         ploc="New York, NY", pyears="2018 – 2021",
         edu="B.S. Computer Information Systems, Baruch College, 2018",
         fit="weak"),

    # ── Tier A · Non-local · deep_tier1 ─────────────────────────────────

    dict(name="Nathan Lee",
         email="nathan.lee@gmail.com", location="San Francisco, CA",
         stage="Applied", applied="2025-04-14",
         title="Staff LLMOps & LangGraph Engineer", company="Anthropic",
         cloc="San Francisco, CA", rstart="2021", rend="Present",
         prev_title="ML Platform Engineer", prev_company="OpenAI",
         ploc="San Francisco, CA", pyears="2019 – 2021",
         edu="M.S. Computer Science, Stanford University, 2019; B.S. CS, UC San Diego, 2017",
         fit="deep_tier1"),

    dict(name="Valentina Cruz",
         email="v.cruz@outlook.com", location="Seattle, WA",
         stage="Applied", applied="2025-04-15",
         title="LLM Infrastructure Lead — LangGraph & AWS", company="Microsoft Azure AI",
         cloc="Redmond, WA", rstart="2020", rend="Present",
         prev_title="Cloud Engineer", prev_company="Expedia Group",
         ploc="Seattle, WA", pyears="2017 – 2020",
         edu="M.S. Computer Science, University of Washington, 2017; B.S. CS, UCLA, 2015",
         fit="deep_tier1"),

    dict(name="Oliver Johnson",
         email="o.johnson@gmail.com", location="Austin, TX",
         stage="Applied", applied="2025-04-16",
         title="Senior LLMOps Engineer — LangGraph & AWS", company="Dell AI Labs",
         cloc="Austin, TX", rstart="2022", rend="Present",
         prev_title="MLOps Engineer", prev_company="Indeed.com",
         ploc="Austin, TX", pyears="2020 – 2022",
         edu="B.S. Electrical & Computer Engineering, UT Austin, 2020",
         fit="deep_tier1"),

    # ── Tier A · Non-local · medium_tier1 ───────────────────────────────

    dict(name="Maya Williams",
         email="maya.w@icloud.com", location="Boston, MA",
         stage="Applied", applied="2025-04-17",
         title="ML Platform Engineer", company="Wayfair",
         cloc="Boston, MA", rstart="2021", rend="Present",
         prev_title="Data Engineer", prev_company="HubSpot",
         ploc="Cambridge, MA", pyears="2019 – 2021",
         edu="B.S. Computer Science, Northeastern University, 2019",
         fit="medium_tier1"),

    dict(name="Ethan Park",
         email="ethan.park@gmail.com", location="Seattle, WA",
         stage="Applied", applied="2025-04-18",
         title="ML Engineer", company="Expedia Group",
         cloc="Seattle, WA", rstart="2021", rend="Present",
         prev_title="Software Engineer", prev_company="Zillow Group",
         ploc="Seattle, WA", pyears="2019 – 2021",
         edu="B.S. Computer Science, University of Washington, 2019",
         fit="medium_tier1"),

    # ── Tier A · Non-local · tier2_deep ─────────────────────────────────

    dict(name="Rahul Gupta",
         email="r.gupta@proton.me", location="Chicago, IL",
         stage="Applied", applied="2025-04-19",
         title="MLOps Lead", company="United Airlines",
         cloc="Chicago, IL", rstart="2020", rend="Present",
         prev_title="DevOps Engineer", prev_company="Boeing",
         ploc="Chicago, IL", pyears="2016 – 2020",
         edu="M.S. Computer Science, UIC, 2016; B.E. CS, Mumbai University, 2014",
         fit="tier2_deep"),

    dict(name="Jasmine Washington",
         email="jas.washington@gmail.com", location="Denver, CO",
         stage="Applied", applied="2025-04-20",
         title="Senior DevOps & AI Infrastructure Engineer", company="LogicWorks",
         cloc="Denver, CO", rstart="2022", rend="Present",
         prev_title="DevOps Engineer", prev_company="Lockheed Martin",
         ploc="Denver, CO", pyears="2019 – 2022",
         edu="B.S. Computer Engineering, University of Colorado Boulder, 2019",
         fit="tier2_deep"),

    dict(name="Lucia Martinez",
         email="lucia.m@icloud.com", location="San Francisco, CA",
         stage="Applied", applied="2025-04-21",
         title="LLMOps Engineer", company="Salesforce",
         cloc="San Francisco, CA", rstart="2022", rend="Present",
         prev_title="MLOps Engineer", prev_company="Tableau",
         ploc="Seattle, WA", pyears="2020 – 2022",
         edu="B.S. Computer Science, UC Berkeley, 2020",
         fit="tier2_deep"),

    # ── Tier A · Non-local · tier2_skills ───────────────────────────────

    dict(name="Daniel Nguyen",
         email="d.nguyen@gmail.com", location="Atlanta, GA",
         stage="Applied", applied="2025-04-22",
         title="DevOps Engineer III", company="NCR Corporation",
         cloc="Atlanta, GA", rstart="2021", rend="Present",
         prev_title="Junior DevOps", prev_company="Cardlytics",
         ploc="Atlanta, GA", pyears="2019 – 2021",
         edu="B.S. Computer Science, Georgia Tech, 2019",
         fit="tier2_skills"),

    dict(name="Alex Turner",
         email="a.turner@outlook.com", location="Dallas, TX",
         stage="Applied", applied="2025-04-23",
         title="Senior Site Reliability Engineer", company="AT&T",
         cloc="Dallas, TX", rstart="2020", rend="Present",
         prev_title="Systems Engineer", prev_company="Southwest Airlines",
         ploc="Dallas, TX", pyears="2017 – 2020",
         edu="B.S. Computer Science, UT Dallas, 2017",
         fit="tier2_skills"),

    dict(name="Anika Singh",
         email="anika.singh@gmail.com", location="Raleigh, NC",
         stage="Applied", applied="2025-04-24",
         title="Platform Engineer", company="Red Hat",
         cloc="Raleigh, NC", rstart="2021", rend="Present",
         prev_title="Systems Administrator", prev_company="Cisco",
         ploc="Research Triangle, NC", pyears="2018 – 2021",
         edu="B.S. Information Technology, NC State University, 2018",
         fit="tier2_skills"),

    # ── Tier A · Non-local · general_ml ─────────────────────────────────

    dict(name="Fatima Hassan",
         email="fatima.h@proton.me", location="Los Angeles, CA",
         stage="Applied", applied="2025-04-25",
         title="Data Scientist", company="Disney Streaming",
         cloc="Santa Monica, CA", rstart="2022", rend="Present",
         prev_title="Junior Data Scientist", prev_company="NBCUniversal",
         ploc="Universal City, CA", pyears="2020 – 2022",
         edu="M.S. Data Science, UCLA, 2020; B.S. Statistics, USC, 2018",
         fit="general_ml"),

    dict(name="Simone Adeyemi",
         email="s.adeyemi@icloud.com", location="Miami, FL",
         stage="Applied", applied="2025-04-26",
         title="ML Research Engineer", company="Royal Caribbean",
         cloc="Miami, FL", rstart="2022", rend="Present",
         prev_title="Data Scientist", prev_company="Carnival Corp",
         ploc="Miami, FL", pyears="2020 – 2022",
         edu="M.S. Computer Science, Florida International University, 2020; B.S. CS, University of Miami, 2018",
         fit="general_ml"),

    dict(name="Marcus Webb",
         email="marcus.webb@gmail.com", location="Nashville, TN",
         stage="Applied", applied="2025-04-27",
         title="ML Engineer", company="HCA Healthcare",
         cloc="Nashville, TN", rstart="2022", rend="Present",
         prev_title="Data Analyst", prev_company="Community Health Systems",
         ploc="Franklin, TN", pyears="2020 – 2022",
         edu="B.S. Computer Science, Vanderbilt University, 2020",
         fit="general_ml"),

    # ── Tier A · Non-local · weak ────────────────────────────────────────

    dict(name="Patrick Murphy",
         email="p.murphy@gmail.com", location="Portland, OR",
         stage="Applied", applied="2025-04-28",
         title="Backend Lead", company="Nike Technology",
         cloc="Beaverton, OR", rstart="2021", rend="Present",
         prev_title="Software Engineer", prev_company="Columbia Sportswear",
         ploc="Portland, OR", pyears="2018 – 2021",
         edu="B.S. Computer Science, University of Oregon, 2018",
         fit="weak"),

    dict(name="Hannah Chen",
         email="hannah.chen@icloud.com", location="Phoenix, AZ",
         stage="Applied", applied="2025-04-29",
         title="Software Architect", company="Avnet",
         cloc="Phoenix, AZ", rstart="2019", rend="Present",
         prev_title="Senior Engineer", prev_company="Intel",
         ploc="Chandler, AZ", pyears="2015 – 2019",
         edu="B.S. Computer Engineering, Arizona State University, 2015",
         fit="weak"),

    dict(name="Jerome Davis",
         email="j.davis@proton.me", location="Houston, TX",
         stage="Applied", applied="2025-04-30",
         title="Cloud Engineer", company="ExxonMobil",
         cloc="Spring, TX", rstart="2021", rend="Present",
         prev_title="Systems Engineer", prev_company="Shell",
         ploc="Houston, TX", pyears="2018 – 2021",
         edu="B.S. Computer Science, University of Houston, 2018",
         fit="weak"),

    dict(name="Sarah Kim",
         email="sarah.kim@gmail.com", location="Minneapolis, MN",
         stage="Applied", applied="2025-05-01",
         title="Data Engineer", company="Target Corporation",
         cloc="Minneapolis, MN", rstart="2020", rend="Present",
         prev_title="ETL Developer", prev_company="Best Buy",
         ploc="Richfield, MN", pyears="2017 – 2020",
         edu="B.S. Computer Science, University of Minnesota, 2017",
         fit="weak"),

    # ═══════════════════════════════════════════════════════════════════════
    # TIER B — most recent role < 3 years, degree ≤ 2022 (no recent degree)
    # ═══════════════════════════════════════════════════════════════════════

    # ── Tier B · NY Metro ────────────────────────────────────────────────

    dict(name="Derek Chang",
         email="derek.chang@gmail.com", location="New York, NY",
         stage="Phone Screen", applied="2025-05-02",
         title="LLMOps & LangGraph Engineer", company="Nielsen",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="ML Engineer", prev_company="Moody's",
         ploc="New York, NY", pyears="2021 – 2024",
         edu="B.S. Computer Science, Fordham University, 2021",
         fit="deep_tier1"),

    dict(name="Natasha Ivanova",
         email="n.ivanova@icloud.com", location="Brooklyn, NY",
         stage="Applied", applied="2025-05-03",
         title="AI Engineer", company="Squarespace",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="ML Engineer", prev_company="Etsy",
         ploc="Brooklyn, NY", pyears="2021 – 2024",
         edu="M.S. Data Science, NYU, 2021; B.S. Mathematics, Stony Brook University, 2019",
         fit="medium_tier1"),

    dict(name="Ryan Kowalski",
         email="ryan.k@outlook.com", location="Hoboken, NJ",
         stage="Applied", applied="2025-05-04",
         title="MLOps Engineer", company="Chubb Insurance",
         cloc="Jersey City, NJ", rstart="2024", rend="Present",
         prev_title="DevOps Engineer", prev_company="Cognizant",
         ploc="Teaneck, NJ", pyears="2021 – 2024",
         edu="B.S. Information Technology, NJIT, 2021",
         fit="tier2_deep"),

    dict(name="Elena Vasquez",
         email="elena.v@gmail.com", location="Jersey City, NJ",
         stage="Applied", applied="2025-05-05",
         title="DevOps Engineer", company="EY Technology",
         cloc="New York, NY", rstart="2025", rend="Present",
         prev_title="Junior DevOps", prev_company="Deloitte",
         ploc="New York, NY", pyears="2022 – 2025",
         edu="B.S. Computer Science, Montclair State University, 2022",
         fit="tier2_skills"),

    dict(name="Brandon Harris",
         email="b.harris@proton.me", location="Queens, NY",
         stage="Applied", applied="2025-05-06",
         title="Data Scientist", company="AIG",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="Data Analyst", prev_company="MetLife",
         ploc="New York, NY", pyears="2021 – 2024",
         edu="B.S. Statistics, Queens College CUNY, 2021",
         fit="general_ml"),

    dict(name="Jennifer Liu",
         email="jen.liu@gmail.com", location="Brooklyn, NY",
         stage="Applied", applied="2025-05-07",
         title="Software Engineer", company="Peloton",
         cloc="New York, NY", rstart="2025", rend="Present",
         prev_title="Junior Engineer", prev_company="WeWork",
         ploc="New York, NY", pyears="2022 – 2025",
         edu="B.S. Computer Science, Brooklyn College, 2022",
         fit="weak"),

    # ── Tier B · Non-local ───────────────────────────────────────────────

    dict(name="Connor Walsh",
         email="c.walsh@icloud.com", location="Seattle, WA",
         stage="Phone Screen", applied="2025-05-08",
         title="LangGraph & LLMOps Platform Engineer", company="Databricks",
         cloc="Seattle, WA", rstart="2024", rend="Present",
         prev_title="ML Engineer", prev_company="Tableau",
         ploc="Seattle, WA", pyears="2021 – 2024",
         edu="B.S. Computer Science, University of Washington, 2021",
         fit="deep_tier1"),

    dict(name="Nadia Petrov",
         email="nadia.p@gmail.com", location="San Francisco, CA",
         stage="Applied", applied="2025-05-09",
         title="AI Engineer", company="Notion",
         cloc="San Francisco, CA", rstart="2024", rend="Present",
         prev_title="ML Engineer", prev_company="Figma",
         ploc="San Francisco, CA", pyears="2021 – 2024",
         edu="M.S. Computer Science, UC Berkeley, 2021; B.S. CS, UC Davis, 2019",
         fit="medium_tier1"),

    dict(name="Malik Robinson",
         email="malik.r@proton.me", location="Austin, TX",
         stage="Applied", applied="2025-05-10",
         title="ML Infrastructure Engineer", company="Tesla Energy",
         cloc="Austin, TX", rstart="2024", rend="Present",
         prev_title="DevOps Engineer", prev_company="Dell Technologies",
         ploc="Austin, TX", pyears="2020 – 2024",
         edu="B.S. Electrical Engineering, UT Austin, 2020",
         fit="tier2_deep"),

    dict(name="Claire Anderson",
         email="claire.a@outlook.com", location="Chicago, IL",
         stage="Applied", applied="2025-05-11",
         title="DevOps Engineer", company="Accenture",
         cloc="Chicago, IL", rstart="2025", rend="Present",
         prev_title="Systems Administrator", prev_company="Motorola Solutions",
         ploc="Chicago, IL", pyears="2022 – 2025",
         edu="B.S. Computer Engineering, University of Illinois at Chicago, 2022",
         fit="tier2_skills"),

    dict(name="Joshua Brown",
         email="josh.brown@gmail.com", location="Boston, MA",
         stage="Applied", applied="2025-05-12",
         title="ML Engineer", company="athenahealth",
         cloc="Watertown, MA", rstart="2024", rend="Present",
         prev_title="Data Scientist", prev_company="Optum",
         ploc="Boston, MA", pyears="2021 – 2024",
         edu="B.S. Statistics, Boston University, 2021",
         fit="general_ml"),

    dict(name="Anastasia Volkov",
         email="a.volkov@icloud.com", location="Denver, CO",
         stage="Applied", applied="2025-05-13",
         title="AI Platform Engineer", company="Travelers Insurance",
         cloc="Hartford, CT", rstart="2025", rend="Present",
         prev_title="DevOps Engineer", prev_company="USAA",
         ploc="Denver, CO", pyears="2022 – 2025",
         edu="B.S. Computer Information Systems, University of Denver, 2022",
         fit="tier2_skills"),

    dict(name="Kevin Hart",
         email="k.hart@gmail.com", location="Remote",
         stage="Applied", applied="2025-05-14",
         title="Senior Site Reliability Engineer", company="GitLab",
         cloc="Remote", rstart="2024", rend="Present",
         prev_title="SRE", prev_company="PagerDuty",
         ploc="San Francisco, CA", pyears="2020 – 2024",
         edu="B.S. Computer Science, Purdue University, 2020",
         fit="tier2_skills"),

    dict(name="Destiny Jackson",
         email="destiny.j@proton.me", location="Atlanta, GA",
         stage="Applied", applied="2025-05-15",
         title="Data Engineer", company="Cox Media Group",
         cloc="Atlanta, GA", rstart="2024", rend="Present",
         prev_title="Junior Data Engineer", prev_company="Turner Broadcasting",
         ploc="Atlanta, GA", pyears="2021 – 2024",
         edu="B.S. Information Systems, Georgia State University, 2021",
         fit="weak"),

    dict(name="Tyler Ross",
         email="tyler.ross@outlook.com", location="Dallas, TX",
         stage="Applied", applied="2025-05-16",
         title="Backend Developer", company="Sabre Corporation",
         cloc="Southlake, TX", rstart="2025", rend="Present",
         prev_title="Software Engineer", prev_company="American Airlines",
         ploc="Fort Worth, TX", pyears="2021 – 2025",
         edu="B.S. Computer Science, Texas A&M University, 2021",
         fit="weak"),

    # ═══════════════════════════════════════════════════════════════════════
    # TIER C — most recent role < 3 years, degree 2023–2025
    # ═══════════════════════════════════════════════════════════════════════

    # ── Tier C · NY Metro · deep_tier1 ──────────────────────────────────

    dict(name="Michelle Zhao",
         email="m.zhao@gmail.com", location="New York, NY",
         stage="Phone Screen", applied="2025-05-17",
         title="LLMOps & LangGraph Engineer", company="Shutterstock",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="AI Research Intern", prev_company="Adobe Research",
         ploc="New York, NY", pyears="Summer 2023", is_internship=True,
         edu="M.S. Computer Science, NYU Courant, 2024; B.S. CS, University of Rochester, 2022",
         fit="deep_tier1"),

    # ── Tier C · NY Metro · medium_tier1 ────────────────────────────────

    dict(name="Javier Morales",
         email="j.morales@icloud.com", location="Brooklyn, NY",
         stage="Applied", applied="2025-05-18",
         title="AI Infrastructure Engineer", company="Yext",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="Software Engineering Intern", prev_company="Etsy",
         ploc="Brooklyn, NY", pyears="Summer 2023", is_internship=True,
         edu="M.S. Computer Science, Columbia University, 2023; B.S. CS, University of Michigan, 2021",
         fit="medium_tier1"),

    # ── Tier C · NY Metro · tier2_deep ──────────────────────────────────

    dict(name="Alicia Thompson",
         email="alicia.t@gmail.com", location="Queens, NY",
         stage="Applied", applied="2025-05-19",
         title="ML Engineer", company="PagerDuty",
         cloc="New York, NY", rstart="2025", rend="Present",
         prev_title="SRE Intern", prev_company="Datadog",
         ploc="New York, NY", pyears="Summer 2024", is_internship=True,
         edu="B.S. Computer Science, CUNY Baruch, 2024",
         fit="tier2_deep"),

    # ── Tier C · NY Metro · tier2_skills ────────────────────────────────

    dict(name="Chris Yamamoto",
         email="chris.y@proton.me", location="Hoboken, NJ",
         stage="Applied", applied="2025-05-20",
         title="AI Engineer", company="West Monroe Partners",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="Software Engineer Intern", prev_company="Prudential",
         ploc="Newark, NJ", pyears="Summer 2023", is_internship=True,
         edu="M.S. Information Systems, Stevens Institute of Technology, 2023; B.S. CS, Rutgers, 2021",
         fit="tier2_skills"),

    dict(name="Nathan Goldberg",
         email="n.goldberg@gmail.com", location="Brooklyn, NY",
         stage="Applied", applied="2025-05-21",
         title="Junior DevOps Engineer", company="Priceline",
         cloc="Norwalk, CT", rstart="2025", rend="Present",
         prev_title="DevOps Intern", prev_company="Booking Holdings",
         ploc="Norwalk, CT", pyears="Summer 2024", is_internship=True,
         edu="B.S. Computer Science, Brooklyn College, 2024",
         fit="tier2_skills"),

    dict(name="Kevin Lin",
         email="kevin.lin@icloud.com", location="Queens, NY",
         stage="Applied", applied="2025-05-22",
         title="ML Engineer", company="Clarifai",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="Data Science Intern", prev_company="Bloomberg",
         ploc="New York, NY", pyears="Summer 2023", is_internship=True,
         edu="B.S. Computer Science, Queens College CUNY, 2023",
         fit="tier2_skills"),

    # ── Tier C · NY Metro · general_ml ──────────────────────────────────

    dict(name="Samira Ahmed",
         email="samira.a@gmail.com", location="Jersey City, NJ",
         stage="Applied", applied="2025-05-23",
         title="Data Scientist", company="ADP",
         cloc="Roseland, NJ", rstart="2025", rend="Present",
         prev_title="Data Science Intern", prev_company="Prudential Financial",
         ploc="Newark, NJ", pyears="Summer 2024", is_internship=True,
         edu="M.S. Data Science, Stevens Institute of Technology, 2024; B.S. Statistics, Rutgers, 2022",
         fit="general_ml"),

    dict(name="Zoe Richardson",
         email="zoe.r@proton.me", location="New York, NY",
         stage="Applied", applied="2025-05-24",
         title="ML Research Engineer", company="MongoDB",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="Research Intern", prev_company="IBM Research",
         ploc="Yorktown Heights, NY", pyears="Summer 2023", is_internship=True,
         edu="B.S. Computer Science, NYU, 2024",
         fit="general_ml"),

    dict(name="Diana Reyes",
         email="diana.r@gmail.com", location="Brooklyn, NY",
         stage="Applied", applied="2025-05-25",
         title="AI Research Engineer", company="NYU Langone Health",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="Graduate Research Assistant", prev_company="NYU Grossman School of Medicine",
         ploc="New York, NY", pyears="2023 – 2024", is_internship=True,
         edu="M.S. Biomedical Informatics, NYU, 2024; B.S. Computer Science, CUNY Hunter, 2022",
         fit="general_ml"),

    # ── Tier C · NY Metro · weak ─────────────────────────────────────────

    dict(name="Aaliyah Foster",
         email="aaliyah.f@icloud.com", location="New York, NY",
         stage="Applied", applied="2025-05-26",
         title="Software Engineer", company="FanDuel",
         cloc="New York, NY", rstart="2025", rend="Present",
         prev_title="Engineering Intern", prev_company="DraftKings",
         ploc="Boston, MA", pyears="Summer 2024", is_internship=True,
         edu="B.S. Computer Science, Fordham University, 2024",
         fit="weak"),

    dict(name="Leo Williams",
         email="leo.w@gmail.com", location="Long Island City, NY",
         stage="Applied", applied="2025-05-27",
         title="Data Engineer", company="The Trade Desk",
         cloc="New York, NY", rstart="2024", rend="Present",
         prev_title="Data Engineering Intern", prev_company="AdRoll",
         ploc="New York, NY", pyears="Summer 2023", is_internship=True,
         edu="B.S. Information Systems, Stony Brook University, 2024",
         fit="weak"),

    dict(name="Priya Kapoor",
         email="priya.kapoor@outlook.com", location="Hoboken, NJ",
         stage="Applied", applied="2025-05-28",
         title="Software Developer", company="Kyndryl",
         cloc="New York, NY", rstart="2025", rend="Present",
         prev_title="IT Intern", prev_company="IBM",
         ploc="Armonk, NY", pyears="Summer 2024", is_internship=True,
         edu="M.S. Information Systems, Stevens Institute of Technology, 2024; B.Tech CS, VIT University, 2022",
         fit="weak"),

    # ── Tier C · Non-local · deep_tier1 ─────────────────────────────────

    dict(name="Imani Walker",
         email="imani.w@gmail.com", location="San Francisco, CA",
         stage="Phone Screen", applied="2025-05-29",
         title="LangGraph & LLMOps Platform Engineer", company="Weights & Biases",
         cloc="San Francisco, CA", rstart="2024", rend="Present",
         prev_title="ML Intern", prev_company="Hugging Face",
         ploc="San Francisco, CA", pyears="Summer 2023", is_internship=True,
         edu="M.S. Computer Science, Stanford University, 2024; B.S. CS, UC Davis, 2022",
         fit="deep_tier1"),

    # ── Tier C · Non-local · medium_tier1 ───────────────────────────────

    dict(name="Jason Okonkwo",
         email="jason.o@icloud.com", location="Seattle, WA",
         stage="Applied", applied="2025-05-30",
         title="ML Infrastructure Engineer", company="Amazon AI",
         cloc="Seattle, WA", rstart="2024", rend="Present",
         prev_title="SDE Intern", prev_company="Amazon",
         ploc="Seattle, WA", pyears="Summer 2023", is_internship=True,
         edu="M.S. Computer Science, University of Washington, 2023; B.S. CS, Purdue, 2021",
         fit="medium_tier1"),

    dict(name="Rishi Patel",
         email="rishi.p@gmail.com", location="Raleigh, NC",
         stage="Applied", applied="2025-05-31",
         title="AI Engineer", company="Bandwidth",
         cloc="Research Triangle, NC", rstart="2024", rend="Present",
         prev_title="Software Engineer Intern", prev_company="Red Hat",
         ploc="Raleigh, NC", pyears="Summer 2023", is_internship=True,
         edu="M.S. Computer Science, NC State University, 2023; B.S. CS, VCU, 2021",
         fit="medium_tier1"),

    # ── Tier C · Non-local · tier2_deep ─────────────────────────────────

    dict(name="Sarah Chang",
         email="sarah.chang@proton.me", location="Austin, TX",
         stage="Applied", applied="2025-06-01",
         title="AI Engineer", company="Siemens EDA",
         cloc="Austin, TX", rstart="2025", rend="Present",
         prev_title="DevOps Intern", prev_company="NXP Semiconductors",
         ploc="Austin, TX", pyears="Summer 2024", is_internship=True,
         edu="B.S. Electrical & Computer Engineering, UT Austin, 2024",
         fit="tier2_deep"),

    dict(name="Xiomara Diaz",
         email="x.diaz@icloud.com", location="San Diego, CA",
         stage="Applied", applied="2025-06-02",
         title="AI Platform Engineer", company="Qualcomm",
         cloc="San Diego, CA", rstart="2024", rend="Present",
         prev_title="SWE Intern", prev_company="Broadcom",
         ploc="San Jose, CA", pyears="Summer 2023", is_internship=True,
         edu="M.S. Computer Science, UC San Diego, 2023; B.S. ECE, UC Santa Barbara, 2021",
         fit="tier2_deep"),

    # ── Tier C · Non-local · tier2_skills ───────────────────────────────

    dict(name="Mohammed Al-Hassan",
         email="m.alhassan@gmail.com", location="Chicago, IL",
         stage="Applied", applied="2025-06-03",
         title="Junior MLOps Engineer", company="Caterpillar",
         cloc="Peoria, IL", rstart="2024", rend="Present",
         prev_title="DevOps Intern", prev_company="John Deere",
         ploc="Moline, IL", pyears="Summer 2023", is_internship=True,
         edu="M.S. Computer Science, Northwestern University, 2023; B.S. CS, University of Illinois, 2021",
         fit="tier2_skills"),

    dict(name="Diego Hernandez",
         email="diego.h@outlook.com", location="Denver, CO",
         stage="Applied", applied="2025-06-04",
         title="DevOps Engineer", company="Ball Aerospace",
         cloc="Westminster, CO", rstart="2024", rend="Present",
         prev_title="IT Intern", prev_company="Lockheed Martin",
         ploc="Littleton, CO", pyears="Summer 2023", is_internship=True,
         edu="B.S. Computer Science, University of Colorado Boulder, 2024",
         fit="tier2_skills"),

    dict(name="Jordan Maxwell",
         email="jordan.max@gmail.com", location="St. Louis, MO",
         stage="Applied", applied="2025-06-05",
         title="ML Ops Engineer", company="Boeing AI",
         cloc="St. Louis, MO", rstart="2025", rend="Present",
         prev_title="SWE Intern", prev_company="Emerson Electric",
         ploc="Ferguson, MO", pyears="Summer 2024", is_internship=True,
         edu="B.S. Computer Engineering, Washington University in St. Louis, 2024",
         fit="tier2_skills"),

    # ── Tier C · Non-local · general_ml ─────────────────────────────────

    dict(name="Megan Flores",
         email="megan.f@icloud.com", location="Boston, MA",
         stage="Applied", applied="2025-06-06",
         title="ML Engineer", company="Veritas Technologies",
         cloc="Burlington, MA", rstart="2025", rend="Present",
         prev_title="Research Intern", prev_company="MIT CSAIL",
         ploc="Cambridge, MA", pyears="Summer 2024", is_internship=True,
         edu="M.S. Artificial Intelligence, Boston University, 2024; B.S. CS, Northeastern, 2022",
         fit="general_ml"),

    dict(name="Chloe Martin",
         email="chloe.m@proton.me", location="Atlanta, GA",
         stage="Applied", applied="2025-06-07",
         title="Data Scientist", company="NCR Voyix",
         cloc="Atlanta, GA", rstart="2025", rend="Present",
         prev_title="Data Science Intern", prev_company="Chick-fil-A Technology",
         ploc="Atlanta, GA", pyears="Summer 2024", is_internship=True,
         edu="M.S. Data Science, Georgia Tech, 2024; B.S. Statistics, Emory University, 2022",
         fit="general_ml"),

    dict(name="Isaiah Thompson",
         email="isaiah.t@gmail.com", location="Los Angeles, CA",
         stage="Applied", applied="2025-06-08",
         title="ML Engineer", company="Riot Games",
         cloc="Los Angeles, CA", rstart="2024", rend="Present",
         prev_title="ML Research Intern", prev_company="Sony Pictures",
         ploc="Culver City, CA", pyears="Summer 2023", is_internship=True,
         edu="M.S. Computer Science, USC, 2024; B.S. CS, UCLA, 2022",
         fit="general_ml"),

    dict(name="Nour Khalil",
         email="nour.k@icloud.com", location="Miami, FL",
         stage="Applied", applied="2025-06-09",
         title="AI Research Engineer", company="Carnival Corporation",
         cloc="Miami, FL", rstart="2024", rend="Present",
         prev_title="Research Intern", prev_company="University of Miami AI Lab",
         ploc="Coral Gables, FL", pyears="2023 – 2024", is_internship=True,
         edu="M.S. Computer Science, University of Miami, 2024; B.S. CS, Florida Atlantic University, 2022",
         fit="general_ml"),

    # ── Tier C · Non-local · weak ────────────────────────────────────────

    dict(name="Emma Sullivan",
         email="emma.s@gmail.com", location="Portland, OR",
         stage="Applied", applied="2025-06-10",
         title="Junior Software Engineer", company="Daimler Trucks",
         cloc="Portland, OR", rstart="2025", rend="Present",
         prev_title="IT Intern", prev_company="Intel",
         ploc="Hillsboro, OR", pyears="Summer 2024", is_internship=True,
         edu="B.S. Computer Science, Portland State University, 2024",
         fit="weak"),

    dict(name="Marcus Freeman",
         email="marcus.f@proton.me", location="Pittsburgh, PA",
         stage="Applied", applied="2025-06-11",
         title="Software Engineer", company="Duolingo",
         cloc="Pittsburgh, PA", rstart="2024", rend="Present",
         prev_title="SWE Intern", prev_company="PNC Bank",
         ploc="Pittsburgh, PA", pyears="Summer 2023", is_internship=True,
         edu="B.S. Computer Science, Carnegie Mellon University, 2024",
         fit="weak"),

    dict(name="Olivia Chen",
         email="olivia.chen@outlook.com", location="Columbus, OH",
         stage="Applied", applied="2025-06-12",
         title="Associate Data Engineer", company="Nationwide Insurance",
         cloc="Columbus, OH", rstart="2025", rend="Present",
         prev_title="Data Intern", prev_company="Huntington Bank",
         ploc="Columbus, OH", pyears="Summer 2024", is_internship=True,
         edu="B.S. Information Systems, Ohio State University, 2024",
         fit="weak"),

    dict(name="Brennan O'Brien",
         email="brennan.ob@gmail.com", location="Houston, TX",
         stage="Applied", applied="2025-06-13",
         title="Backend Developer", company="Wood Group",
         cloc="Houston, TX", rstart="2025", rend="Present",
         prev_title="SWE Intern", prev_company="Halliburton",
         ploc="Houston, TX", pyears="Summer 2024", is_internship=True,
         edu="B.S. Computer Science, Rice University, 2025",
         fit="weak"),

    dict(name="Tanya Gupta",
         email="tanya.g@icloud.com", location="Phoenix, AZ",
         stage="Applied", applied="2025-06-14",
         title="Data Scientist", company="Banner Health",
         cloc="Phoenix, AZ", rstart="2024", rend="Present",
         prev_title="Data Science Intern", prev_company="Dignity Health",
         ploc="Phoenix, AZ", pyears="Summer 2023", is_internship=True,
         edu="M.S. Biostatistics, Arizona State University, 2024; B.S. Biology, ASU, 2022",
         fit="weak"),

    dict(name="Wei Zhang",
         email="wei.zhang@gmail.com", location="Charlotte, NC",
         stage="Applied", applied="2025-06-15",
         title="Software Engineer", company="Bank of America AI",
         cloc="Charlotte, NC", rstart="2025", rend="Present",
         prev_title="SWE Intern", prev_company="Wells Fargo",
         ploc="Charlotte, NC", pyears="Summer 2024", is_internship=True,
         edu="B.S. Computer Science, UNC Charlotte, 2025",
         fit="weak"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Resume builder
# ─────────────────────────────────────────────────────────────────────────────

_N_VALUES = [7, 8, 10, 12, 15, 20]


def _pick_bullets(fit: str, idx: int) -> list[str]:
    sets = BULLET_SETS[fit]
    chosen = sets[idx % len(sets)]
    n = _N_VALUES[idx % len(_N_VALUES)]
    return [b.format(n=n) for b in chosen]


def build_candidate(spec: dict, idx: int) -> dict:
    fit = spec["fit"]
    years = 2026 - int(spec["rstart"])

    summary = SUMMARY_TEMPLATES[fit].format(title=spec["title"], years=years)

    primary_bullets = _pick_bullets(fit, idx)
    prev_bullets = INTERNSHIP_BULLETS[:2] if spec.get("is_internship") else PREV_BULLETS[fit]

    dates_str = f"{spec['rstart']} – {spec['rend']}"

    # Job title and dates on SAME line so pdfplumber sees a single line with a
    # year → classified as title_line in the experience section.
    # Any keyword in the title then gets depth=10 (title_line + bullets follow).
    title_with_dates = f"{spec['title']}  |  {dates_str}"

    experience = [
        {
            "title": title_with_dates,
            "company": spec["company"],
            "location": spec["cloc"],
            "bullets": primary_bullets,
        },
        {
            "title": spec["prev_title"],
            "company": spec["prev_company"],
            "location": spec["ploc"],
            "dates": spec["pyears"],
            "bullets": prev_bullets,
        },
    ]

    return {
        "name": spec["name"],
        "email": spec["email"],
        "stage": spec["stage"],
        "applied_at": spec["applied"],
        "location": spec["location"],
        "summary": summary,
        "experience": experience,
        "education": spec["edu"],
        "skills": SKILLS_MAP[fit],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PDF generation
# ─────────────────────────────────────────────────────────────────────────────

def make_resume(candidate: dict, output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    navy      = HexColor("#1A1A2E")
    dark_grey = HexColor("#333333")
    mid_grey  = HexColor("#666666")

    name_style = ParagraphStyle(
        "Name", parent=styles["Normal"],
        fontSize=22, textColor=navy, spaceAfter=2,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
    )
    contact_style = ParagraphStyle(
        "Contact", parent=styles["Normal"],
        fontSize=9, textColor=mid_grey, spaceAfter=10,
        fontName="Helvetica", alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Normal"],
        fontSize=11, textColor=navy, spaceBefore=12, spaceAfter=3,
        fontName="Helvetica-Bold",
    )
    summary_style = ParagraphStyle(
        "Summary", parent=styles["Normal"],
        fontSize=10, textColor=dark_grey, spaceAfter=6,
        fontName="Helvetica", leading=14,
    )
    # Title+dates on one line — pdfplumber extracts this as a single string
    # containing the year, so the ranker classifies it as title_line.
    job_title_style = ParagraphStyle(
        "JobTitle", parent=styles["Normal"],
        fontSize=10, textColor=dark_grey, spaceBefore=8, spaceAfter=1,
        fontName="Helvetica-Bold",
    )
    job_meta_style = ParagraphStyle(
        "JobMeta", parent=styles["Normal"],
        fontSize=9, textColor=mid_grey, spaceAfter=3,
        fontName="Helvetica-Oblique",
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=styles["Normal"],
        fontSize=9, textColor=dark_grey, spaceAfter=2,
        fontName="Helvetica", leftIndent=12, leading=13,
        bulletText="•",
    )
    edu_style = ParagraphStyle(
        "Edu", parent=styles["Normal"],
        fontSize=9.5, textColor=dark_grey, spaceAfter=4,
        fontName="Helvetica",
    )
    skills_style = ParagraphStyle(
        "Skills", parent=styles["Normal"],
        fontSize=9.5, textColor=dark_grey, spaceAfter=4,
        fontName="Helvetica",
    )

    divider_color = HexColor("#CCCCCC")
    story = []

    story.append(Paragraph(candidate["name"], name_style))
    story.append(Paragraph(
        f"{candidate['location']}  ·  {candidate['email']}",
        contact_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=navy, spaceAfter=6))

    story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
    story.append(Paragraph(candidate["summary"], summary_style))

    story.append(HRFlowable(width="100%", thickness=0.5, color=divider_color, spaceAfter=2))
    story.append(Paragraph("EXPERIENCE", section_style))

    for job in candidate["experience"]:
        # Primary job: title already includes the dates (title_with_dates).
        # Secondary/prev job: title is plain, dates are on meta line.
        has_inline_dates = "|" in job["title"]
        story.append(Paragraph(job["title"], job_title_style))
        if has_inline_dates:
            story.append(Paragraph(
                f"{job['company']}  |  {job['location']}",
                job_meta_style,
            ))
        else:
            story.append(Paragraph(
                f"{job['company']}  |  {job['location']}  |  {job.get('dates', '')}",
                job_meta_style,
            ))
        for bullet in job["bullets"]:
            story.append(Paragraph(bullet, bullet_style))

    story.append(HRFlowable(width="100%", thickness=0.5, color=divider_color, spaceAfter=2))
    story.append(Paragraph("EDUCATION", section_style))
    story.append(Paragraph(candidate["education"], edu_style))

    story.append(HRFlowable(width="100%", thickness=0.5, color=divider_color, spaceAfter=2))
    story.append(Paragraph("SKILLS", section_style))
    story.append(Paragraph(candidate["skills"], skills_style))

    doc.build(story)


# ─────────────────────────────────────────────────────────────────────────────
# CSV + ZIP
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(candidates: list[dict], path: Path) -> None:
    rows = [
        {
            "Candidate Name": c["name"],
            "Email":          c["email"],
            "Stage":          c["stage"],
            "Applied At":     c["applied_at"],
            "Location":       c["location"],
        }
        for c in candidates
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  ✓ {path}  ({len(rows)} candidates)")


def create_zip(resumes_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pdf in sorted(resumes_dir.glob("*.pdf")):
            zf.write(pdf, pdf.name)
    print(f"  ✓ {zip_path}  ({len(list(resumes_dir.glob('*.pdf')))} PDFs)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generating {len(SPECS)} sample candidates...\n")

    candidates = [build_candidate(s, i) for i, s in enumerate(SPECS)]

    print("Creating sample_resumes/")
    for c in candidates:
        fname = c["name"].replace(" ", "_").replace("'", "") + ".pdf"
        path  = RESUMES_DIR / fname
        make_resume(c, path)
        print(f"  ✓ {fname}")

    print("\nWriting sample_candidates.csv")
    write_csv(candidates, BASE / "sample_candidates.csv")

    print("\nCreating sample_resumes.zip")
    create_zip(RESUMES_DIR, BASE / "sample_resumes.zip")

    # Summary
    tier_counts = {"A": 0, "B": 0, "C": 0}
    ny_count, fit_counts = 0, {}
    ny_metros = {"NY", "NJ", "Brooklyn", "Queens", "Hoboken", "Jersey City", "Manhattan",
                 "Long Island"}
    for s in SPECS:
        # Determine tier from dates and edu
        years = 2026 - int(s["rstart"])
        edu_lower = s["edu"].lower()
        has_recent_deg = any(str(y) in edu_lower for y in range(2023, 2026))
        if years >= 3:
            tier_counts["A"] += 1
        elif has_recent_deg:
            tier_counts["C"] += 1
        else:
            tier_counts["B"] += 1
        if any(m in s["location"] for m in ny_metros):
            ny_count += 1
        fit_counts[s["fit"]] = fit_counts.get(s["fit"], 0) + 1

    total = len(SPECS)
    print(f"\n{'='*60}")
    print(f"  Total: {total} candidates")
    print(f"  Tier A: {tier_counts['A']} ({tier_counts['A']*100//total}%)  "
          f"Tier B: {tier_counts['B']} ({tier_counts['B']*100//total}%)  "
          f"Tier C: {tier_counts['C']} ({tier_counts['C']*100//total}%)")
    print(f"  NY metro: {ny_count} ({ny_count*100//total}%)  "
          f"Non-local: {total-ny_count} ({(total-ny_count)*100//total}%)")
    print(f"  Fit profiles: {fit_counts}")
    print(f"\nRun the ranker:")
    print(f"  python ranker.py sample_candidates.csv sample_resumes/ --jd job.txt "
          f"--output sample_ranked.xlsx")


if __name__ == "__main__":
    main()
