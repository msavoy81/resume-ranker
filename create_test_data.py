#!/usr/bin/env python3
"""
Generate test data for the resume ranker — LLMOps Engineer at Brooklyn Sports & Entertainment.
Creates candidates.csv, resumes/ folder with 10 PDF resumes, and job.txt reference copy.
"""

from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER

BASE = Path(__file__).parent
RESUMES_DIR = BASE / "resumes"
RESUMES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Candidate data — realistic mix for LLMOps Engineer role
# ---------------------------------------------------------------------------

CANDIDATES = [
    # ── TIER 1 — NY-based, strong LLMOps/AWS/Bedrock experience ──────────
    {
        "name": "Marcus Chen",
        "email": "marcus.chen@email.com",
        "stage": "Second Interview",
        "applied_at": "2025-04-01",
        "location": "Brooklyn, NY",
        "summary": (
            "Senior LLMOps Engineer with 7 years of experience building and operating ML and LLM "
            "infrastructure on AWS. Deep hands-on expertise with Amazon Bedrock, LangGraph, and "
            "CI/CD pipelines for AI systems. Proven record delivering production-grade AI platforms "
            "at scale for major media organizations in New York."
        ),
        "experience": [
            {
                "title": "Senior LLMOps Engineer",
                "company": "Paramount+ / CBS Interactive",
                "location": "New York, NY",
                "dates": "2021 – Present",
                "bullets": [
                    "Designed and operates AWS-based LLM infrastructure serving 8 production AI applications across content recommendation, fan engagement, and internal tooling.",
                    "Built CI/CD pipelines for prompt versioning, model deployment, and LangGraph agent graph rollouts using GitHub Actions and AWS CodePipeline.",
                    "Implemented Amazon Bedrock integrations for summarization and personalization features; manages model versioning and automated rollback on SLO breach.",
                    "Built CloudWatch evaluation dashboards tracking hallucination rates, tool-call success, latency p99, and cost-per-request across all LLM systems.",
                    "Manages AWS Secrets Manager, IAM least-privilege policies, and sandboxed agent environments for safe AI-assisted development workflows.",
                ],
            },
            {
                "title": "ML Platform Engineer",
                "company": "Viacom / MTV Networks",
                "location": "New York, NY",
                "dates": "2018 – 2021",
                "bullets": [
                    "Built MLflow-based experiment tracking and model registry for 12 production recommendation models.",
                    "Deployed and operated SageMaker endpoints serving real-time predictions at 50K+ requests/minute.",
                    "Implemented Terraform infrastructure-as-code for the ML platform; reduced provisioning time by 70%.",
                ],
            },
        ],
        "education": "B.S. Computer Science, Columbia University, 2016",
        "skills": (
            "AWS (Bedrock, SageMaker, CloudWatch, CodePipeline, Lambda, ECS, Secrets Manager, IAM), "
            "LangGraph, LangChain, Python, Docker, Kubernetes, Terraform, GitHub Actions, MLflow, "
            "CI/CD, SLO design, observability, infrastructure-as-code"
        ),
    },
    {
        "name": "Priya Sharma",
        "email": "priya.sharma@email.com",
        "stage": "Phone Screen",
        "applied_at": "2025-04-03",
        "location": "New York, NY",
        "summary": (
            "MLOps Engineer specializing in LLM evaluation frameworks and production AI infrastructure. "
            "4 years at JPMorgan Chase building hallucination measurement pipelines and AWS Bedrock "
            "deployment systems. Passionate about AI reliability engineering and agent observability."
        ),
        "experience": [
            {
                "title": "MLOps Engineer",
                "company": "JPMorgan Chase",
                "location": "New York, NY",
                "dates": "2021 – Present",
                "bullets": [
                    "Led production deployment of LLM-based customer service agents using Amazon Bedrock; manages model lifecycle including versioning, evaluation gates, and rollback.",
                    "Built comprehensive evaluation pipeline measuring hallucination rates, response coherence, and workflow reliability across 6 production AI systems.",
                    "Designed automated rollback mechanisms triggered by SLO breaches; reduced mean time to recovery from 45 minutes to under 8 minutes.",
                    "Implemented LangGraph multi-agent workflows for financial analysis tools; established governance guardrails limiting agent permissions.",
                    "Owns CloudWatch and Grafana observability stack; defined 23 SLOs covering latency, error rate, and cost-per-request targets.",
                ],
            },
            {
                "title": "Junior ML Engineer",
                "company": "Capital One",
                "location": "New York, NY",
                "dates": "2019 – 2021",
                "bullets": [
                    "Built AWS Glue and Step Functions pipelines for credit risk model feature engineering.",
                    "Supported SageMaker-based model deployment infrastructure for fraud detection models.",
                ],
            },
        ],
        "education": "M.S. Computer Science, New York University (Courant), 2019; B.Tech Computer Engineering, IIT Delhi, 2017",
        "skills": (
            "AWS (Bedrock, SageMaker, Lambda, Step Functions, CloudWatch, Glue), Python, "
            "LangGraph, LangChain, MLflow, CI/CD, Docker, Terraform, evaluation frameworks, "
            "Grafana, SLO design, incident response"
        ),
    },
    {
        "name": "Jordan Williams",
        "email": "jordan.williams@email.com",
        "stage": "First Interview",
        "applied_at": "2025-04-05",
        "location": "Long Island City, NY",
        "summary": (
            "ML Infrastructure Lead with 8 years of experience, transitioning from SRE to ML platform "
            "engineering. Deep AWS expertise, production reliability background, and hands-on experience "
            "integrating Amazon Bedrock into consumer-facing features at Spotify's New York office."
        ),
        "experience": [
            {
                "title": "ML Infrastructure Lead",
                "company": "Spotify",
                "location": "Long Island City, NY",
                "dates": "2020 – Present",
                "bullets": [
                    "Designs and operates AWS ML platform serving 10+ production models across recommendation, personalization, and podcast audio analysis.",
                    "Integrated Amazon Bedrock for podcast summarization and content discovery; owns prompt versioning and deployment pipelines.",
                    "Built CI/CD pipelines for model deployment, A/B testing rollouts, and canary releases; maintains 99.97% uptime SLO.",
                    "Implements CloudWatch and Datadog observability; reduced ML infrastructure spend 25% through cost-per-request optimization.",
                    "Leads incident response for AI system outages; authored operational playbooks reducing MTTR by 40%.",
                ],
            },
            {
                "title": "Senior Site Reliability Engineer",
                "company": "Spotify",
                "location": "New York, NY",
                "dates": "2015 – 2020",
                "bullets": [
                    "Production reliability engineering for Spotify's core streaming infrastructure on AWS (EKS, EC2, RDS, S3).",
                    "Built self-healing infrastructure and automated runbooks; handled 100M+ daily active users.",
                ],
            },
        ],
        "education": "B.S. Computer Science, Stony Brook University, 2015",
        "skills": (
            "AWS (Bedrock, EKS, SageMaker, CloudWatch, EC2, Lambda, S3), Python, Go, "
            "Terraform, Kubernetes, Datadog, CI/CD, SLO design, MLflow, reliability engineering, "
            "incident response, infrastructure-as-code"
        ),
    },
    # ── TIER 2 — Right skills but outside NY, or partial fit ─────────────
    {
        "name": "Alex Torres",
        "email": "alex.torres@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-07",
        "location": "San Francisco, CA",
        "summary": (
            "Staff MLOps Engineer with 7 years specializing in LLM infrastructure at AI-native companies. "
            "Deep expertise in Amazon Bedrock, evaluation frameworks, and multi-cloud LLM deployment pipelines. "
            "Seeking to relocate; no current NY connections."
        ),
        "experience": [
            {
                "title": "Staff MLOps Engineer",
                "company": "Cohere",
                "location": "San Francisco, CA",
                "dates": "2020 – Present",
                "bullets": [
                    "Leads MLOps infrastructure for enterprise customers deploying Cohere models on AWS and Azure; manages 30+ production LLM environments.",
                    "Designed multi-environment CI/CD pipeline (dev/staging/prod) with automated evaluation gates measuring hallucination, coherence, and task completion.",
                    "Built prompt versioning system and automated rollback pipelines; reduced production incidents by 60% in 18 months.",
                    "Works daily with Amazon Bedrock, LangGraph, and custom evaluation frameworks for agentic workflows.",
                    "Implements secrets management, IAM policies, and sandboxed agent environments per enterprise security requirements.",
                ],
            },
            {
                "title": "MLOps Engineer",
                "company": "Weights & Biases",
                "location": "San Francisco, CA",
                "dates": "2018 – 2020",
                "bullets": [
                    "Built CI/CD integrations enabling automated experiment tracking and model evaluation at 500+ enterprise customers.",
                    "Developed automated model evaluation pipelines; contributed to open-source MLOps tooling.",
                ],
            },
        ],
        "education": "B.S. Computer Science, UC Berkeley, 2017",
        "skills": (
            "AWS (Bedrock, SageMaker, ECS, Lambda, CloudWatch, CodePipeline), Azure OpenAI, "
            "LangGraph, LangChain, Python, Terraform, Docker, CI/CD, evaluation frameworks, "
            "MLflow, GitHub Actions, secrets management"
        ),
    },
    {
        "name": "Kevin Nguyen",
        "email": "kevin.nguyen@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-08",
        "location": "Hoboken, NJ",
        "summary": (
            "ML Engineer at Bloomberg with 4 years of experience building data pipelines and ML infrastructure "
            "in New York's financial sector. Growing hands-on experience with LLM integration via AWS Bedrock "
            "and LangChain; strong Python and AWS foundations."
        ),
        "experience": [
            {
                "title": "ML Engineer",
                "company": "Bloomberg L.P.",
                "location": "Manhattan, NY",
                "dates": "2022 – Present",
                "bullets": [
                    "Builds and maintains ML infrastructure for Bloomberg's financial analytics platform on AWS.",
                    "Recently integrated Amazon Bedrock for an internal news summarization feature; building out CI/CD pipeline for prompt versioning.",
                    "Works with LangChain for early-stage agent workflows supporting research tools.",
                    "Manages AWS infrastructure (ECS, Lambda, S3, CloudWatch) for ML model serving.",
                ],
            },
            {
                "title": "Software Engineer",
                "company": "Fiserv",
                "location": "Jersey City, NJ",
                "dates": "2021 – 2022",
                "bullets": [
                    "Backend Python/Java development for financial services APIs on AWS (S3, Lambda, EC2).",
                    "Maintained CI/CD pipelines using GitHub Actions for microservices deployments.",
                ],
            },
        ],
        "education": "B.S. Software Engineering, Rutgers University, 2020",
        "skills": (
            "Python, AWS (EC2, Lambda, ECS, S3, CloudWatch, Bedrock — learning), "
            "LangChain, GitHub Actions, Docker, SQL, scikit-learn, basic MLOps"
        ),
    },
    {
        "name": "Sarah Mitchell",
        "email": "sarah.mitchell@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-09",
        "location": "Chicago, IL",
        "summary": (
            "Senior DevOps Engineer with 6 years of deep CI/CD and AWS infrastructure expertise. "
            "Strong background in production reliability, Kubernetes, and observability — but experience "
            "is in traditional software DevOps, not ML or AI systems. Open to relocating."
        ),
        "experience": [
            {
                "title": "Senior DevOps Engineer",
                "company": "Motorola Solutions",
                "location": "Chicago, IL",
                "dates": "2021 – Present",
                "bullets": [
                    "Manages CI/CD pipelines for firmware and software releases across 200+ microservices using GitHub Actions, Jenkins, and AWS CodePipeline.",
                    "AWS infrastructure management (EC2, ECS, RDS, CloudWatch, Route 53) for SaaS products serving enterprise public safety customers.",
                    "Implemented Terraform IaC and Kubernetes orchestration; achieved 99.95% uptime SLO across production fleet.",
                    "Built CloudWatch monitoring and alerting dashboards; trained 15 engineers on observability best practices.",
                ],
            },
            {
                "title": "DevOps Engineer",
                "company": "Zebra Technologies",
                "location": "Chicago, IL",
                "dates": "2019 – 2021",
                "bullets": [
                    "Automated deployment pipelines reducing deploy cycle time from 4 hours to under 20 minutes.",
                    "Managed AWS infrastructure for enterprise IoT platform; introduced Docker containerization.",
                ],
            },
        ],
        "education": "B.S. Computer Engineering, University of Illinois Urbana-Champaign, 2019",
        "skills": (
            "AWS (EC2, ECS, RDS, CloudWatch, CodePipeline, Route 53), Kubernetes, Terraform, "
            "GitHub Actions, Jenkins, Docker, Python, CI/CD, observability — no ML/LLM experience"
        ),
    },
    # ── RECENT GRADS — NY local, short tenure, recent degree ─────────────
    {
        "name": "Daniel Reyes",
        "email": "daniel.reyes@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-14",
        "location": "Queens, NY",
        "summary": (
            "MLOps Engineer with strong hands-on skills in LLM infrastructure and AWS. "
            "Built production-grade agentic pipelines using LangGraph and Amazon Bedrock during "
            "graduate research and a post-graduation role at a Manhattan AI startup. M.S. in "
            "Computer Science from Columbia University, completed May 2024."
        ),
        "experience": [
            {
                "title": "MLOps Engineer",
                "company": "Mosaic AI (startup)",
                "location": "New York, NY",
                "dates": "2024 – Present",
                "bullets": [
                    "Builds LangGraph-based multi-agent workflows for enterprise document analysis; deploys on AWS ECS with CI/CD pipelines managed via GitHub Actions.",
                    "Integrated Amazon Bedrock (Claude and Titan models) for summarization and extraction tasks; owns prompt versioning and evaluation pipelines.",
                    "Implements MLOps observability using CloudWatch and custom LLM evaluation metrics; monitors hallucination rates and latency p95.",
                    "Manages Kubernetes clusters for model serving infrastructure; automated blue-green deployments reducing downtime to zero.",
                ],
            },
            {
                "title": "Graduate Research Assistant — ML Systems Lab",
                "company": "Columbia University",
                "location": "New York, NY",
                "dates": "2022 – 2024",
                "bullets": [
                    "Developed RAG pipelines and LLM evaluation benchmarks for academic research on agentic systems.",
                    "Published paper on LLM observability tooling; built open-source MLflow-based evaluation framework.",
                ],
            },
        ],
        "education": "M.S. Computer Science, Columbia University, May 2024; B.S. Computer Science, NYU, 2022",
        "skills": (
            "AWS (Bedrock, ECS, Lambda, CloudWatch, S3), LangGraph, LangChain, Python, "
            "Kubernetes, Docker, GitHub Actions, CI/CD, MLflow, RAG, LLM evaluation, "
            "observability, agentic workflows, Generative AI"
        ),
    },
    {
        "name": "Natalie Russo",
        "email": "natalie.russo@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-15",
        "location": "Astoria, NY",
        "summary": (
            "Junior software engineer from Queens with a B.S. in Information Technology completed in 2023. "
            "One year of professional experience in backend development with some exposure to AWS and basic "
            "machine learning. Interested in transitioning toward AI/ML infrastructure."
        ),
        "experience": [
            {
                "title": "Software Engineer",
                "company": "Revit Health (startup)",
                "location": "New York, NY",
                "dates": "2024 – Present",
                "bullets": [
                    "Builds Python and Flask backend APIs for a healthcare data platform; deploys on AWS EC2 and S3.",
                    "Completed an internal workshop on AWS SageMaker; no production ML deployments yet.",
                    "Uses basic CI/CD with GitHub Actions for automated testing and deployments.",
                ],
            },
            {
                "title": "IT Support Intern",
                "company": "NYC Department of Education",
                "location": "New York, NY",
                "dates": "Summer 2022",
                "bullets": [
                    "Provided technical support and assisted with basic network administration tasks.",
                ],
            },
        ],
        "education": "B.S. Information Technology, CUNY Queens College, May 2023",
        "skills": (
            "Python, Flask, REST APIs, basic AWS (EC2, S3), GitHub Actions, SQL, "
            "HTML/CSS, no MLOps, no LLM, no DevOps experience"
        ),
    },
    # ── TIER 3 — Underqualified, wrong skill set, or job hoppers ─────────
    {
        "name": "Emma Park",
        "email": "emma.park@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-10",
        "location": "Brooklyn, NY",
        "summary": (
            "Junior software engineer based in Brooklyn with 1 year of professional experience. "
            "Enthusiastic about AI and cloud technologies; currently self-studying AWS and machine learning. "
            "No production MLOps, LLM, or DevOps experience yet."
        ),
        "experience": [
            {
                "title": "Junior Software Engineer",
                "company": "Etsy",
                "location": "Brooklyn, NY",
                "dates": "2025 – Present",
                "bullets": [
                    "Builds Python REST APIs for Etsy's seller analytics platform.",
                    "Uses basic AWS services (S3, Lambda) for data storage; currently learning CloudWatch.",
                    "Attending internal AI knowledge-sharing sessions; completed AWS Cloud Practitioner certification.",
                ],
            },
            {
                "title": "Software Engineering Intern",
                "company": "NYC Digital (City of New York)",
                "location": "New York, NY",
                "dates": "Summer 2024",
                "bullets": [
                    "Developed REST APIs for a city services portal; assisted with basic data engineering.",
                ],
            },
        ],
        "education": "B.S. Computer Science, CUNY Brooklyn College, 2024",
        "skills": "Python, JavaScript, basic AWS (S3, Lambda), SQL, REST APIs, Git",
    },
    {
        "name": "Tyler Brooks",
        "email": "tyler.brooks@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-11",
        "location": "Austin, TX",
        "summary": (
            "Data scientist with Python and analytics experience across several short-term roles. "
            "Familiar with basic ML concepts and data analysis; limited cloud infrastructure or "
            "DevOps experience. No production LLM or MLOps background."
        ),
        "experience": [
            {
                "title": "Data Scientist",
                "company": "Dropbox",
                "location": "Austin, TX (Remote)",
                "dates": "2025 – Present",
                "bullets": [
                    "Builds analytics models in Python for product usage data; 6 months in role.",
                ],
            },
            {
                "title": "ML Engineer",
                "company": "Clearbit",
                "location": "Austin, TX",
                "dates": "2024 – 2025",
                "bullets": [
                    "Trained classification models for B2B data enrichment; used basic AWS S3 for data storage.",
                ],
            },
            {
                "title": "Data Analyst",
                "company": "Dell Technologies",
                "location": "Austin, TX",
                "dates": "2023 – 2024",
                "bullets": [
                    "SQL reporting and Tableau dashboards for sales operations.",
                ],
            },
            {
                "title": "Junior Data Scientist",
                "company": "Local AI Startup",
                "location": "Austin, TX",
                "dates": "2022 – 2023",
                "bullets": [
                    "Exploratory data analysis and basic NLP experiments using Python and Jupyter notebooks.",
                ],
            },
        ],
        "education": "B.S. Statistics, University of Texas at Austin, 2021",
        "skills": "Python, pandas, scikit-learn, SQL, Tableau, Jupyter, basic AWS (S3 only), no DevOps/MLOps",
    },
    {
        "name": "Aisha Williams",
        "email": "aisha.williams@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-12",
        "location": "Boston, MA",
        "summary": (
            "Senior data analyst with strong SQL, Tableau, and business intelligence skills at a major "
            "financial services firm. No software engineering, cloud infrastructure, or AI/ML systems "
            "experience. Interested in pivoting toward a more technical role."
        ),
        "experience": [
            {
                "title": "Senior Data Analyst",
                "company": "Fidelity Investments",
                "location": "Boston, MA",
                "dates": "2022 – Present",
                "bullets": [
                    "Builds Tableau and Power BI dashboards for portfolio analytics consumed by 200+ advisors.",
                    "Writes complex SQL queries against enterprise data warehouse; some Python for data manipulation.",
                    "Collaborates with engineering teams on data requirements; no infrastructure ownership.",
                ],
            },
            {
                "title": "Data Analyst",
                "company": "State Street",
                "location": "Boston, MA",
                "dates": "2021 – 2022",
                "bullets": [
                    "Reporting and business intelligence for institutional asset management division.",
                ],
            },
        ],
        "education": "B.S. Information Systems, Boston University, 2021",
        "skills": "SQL, Tableau, Power BI, Excel, Python (basic scripts), no AWS, no ML engineering, no DevOps",
    },
    {
        "name": "Chris Johnson",
        "email": "chris.johnson@email.com",
        "stage": "Applied",
        "applied_at": "2025-04-13",
        "location": "Phoenix, AZ",
        "summary": (
            "Backend web developer with 1 year of experience building REST APIs. Completed a software "
            "engineering bootcamp in 2024. Basic familiarity with AWS compute and storage services; "
            "no ML, AI, DevOps, or infrastructure engineering experience."
        ),
        "experience": [
            {
                "title": "Backend Developer",
                "company": "Mesa Digital",
                "location": "Phoenix, AZ (Remote)",
                "dates": "2025 – Present",
                "bullets": [
                    "Builds REST APIs in Node.js and Python for a local government services platform.",
                    "Basic AWS deployment (EC2, S3) for web application hosting; no DevOps ownership.",
                ],
            },
            {
                "title": "Software Engineering Intern",
                "company": "Phoenix Tech Collective",
                "location": "Phoenix, AZ",
                "dates": "Summer 2024",
                "bullets": [
                    "Built CRUD features in a Django web application during bootcamp capstone project.",
                ],
            },
        ],
        "education": "Fullstack Academy — Software Engineering Bootcamp, 2024",
        "skills": "JavaScript, Node.js, Python, REST APIs, basic SQL, basic AWS (EC2, S3), Git — no ML or DevOps",
    },
]


# ---------------------------------------------------------------------------
# PDF creation
# ---------------------------------------------------------------------------

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
    navy = HexColor("#1A1A2E")
    dark_grey = HexColor("#333333")
    mid_grey = HexColor("#666666")

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
        story.append(Paragraph(job["title"], job_title_style))
        story.append(Paragraph(
            f"{job['company']}  |  {job['location']}  |  {job['dates']}",
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


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def write_csv() -> None:
    rows = [
        {
            "Candidate Name": c["name"],
            "Email": c["email"],
            "Stage": c["stage"],
            "Applied At": c["applied_at"],
            "Location": c["location"],
        }
        for c in CANDIDATES
    ]
    out = BASE / "candidates.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  ✓ {out} ({len(rows)} candidates)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating test data for LLMOps Engineer role...\n")

    print("Creating resumes/")
    for c in CANDIDATES:
        fname = c["name"].replace(" ", "_") + ".pdf"
        path = RESUMES_DIR / fname
        make_resume(c, path)
        print(f"  ✓ {path.name}")

    print("\nWriting candidates.csv")
    write_csv()

    print(f"\nDone! {len(CANDIDATES)} candidates generated.")
    print("  Tier 1 (NY + LLMOps): Marcus Chen, Priya Sharma, Jordan Williams")
    print("  Tier 2 (partial fit): Alex Torres, Kevin Nguyen, Sarah Mitchell")
    print("  Recent grads (Tier B, should rank bottom): Daniel Reyes, Natalie Russo")
    print("  Tier 3 (underqualified): Emma Park, Tyler Brooks, Aisha Williams, Chris Johnson")


if __name__ == "__main__":
    main()
