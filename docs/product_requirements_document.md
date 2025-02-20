AI-Driven Candidate Matcher (MVP) – Product Requirements Document

1. Scope and Objectives

Product Vision: Develop an AI-driven platform that intelligently matches job postings with candidate profiles (and vice versa) using advanced language models and semantic search. This MVP will serve both recruiters and job seekers, providing a fully functional, hostable solution that can be deployed in real-world recruitment scenarios.

Objectives:
	•	End-to-End Matching: Allow recruiters to input a job description and find the best candidate matches, and allow job seekers to input their resume or profile to discover relevant job openings. Both flows should be intuitive and yield useful, ranked results.
	•	Core AI Functionality: Leverage vector embeddings and retrieval-augmented generation (RAG) to enable semantic matching and detailed AI-generated explanations. RAG will provide the language model with relevant context (e.g. candidate resumes or job descriptions from a database) at query time to improve accuracy. This approach mitigates common LLM limitations (such as outdated knowledge or lack of private domain data) by injecting up-to-date, domain-specific information into the matching process.
	•	Usability: Ensure the system is usable by non-technical users. Recruiters should be able to use a simple interface or API to get candidate recommendations for a role, and job seekers should easily find jobs matching their profile. The focus is on delivering clear value (good matches with explanations) without requiring understanding of the underlying AI. Large Language Models (LLMs) are capable of handling tasks like resume screening and job matching ￼, so this MVP aims to harness that capability in a user-friendly way.
	•	Production-Ready Deployment: The MVP must be deployable in a production environment. It should be fully functional and hostable on cloud infrastructure (AWS), following best practices for scalability and reliability. That includes using serverless components where appropriate and ensuring the design can handle real-world data sizes (initially small-scale, but with a path to scaling up).
	•	Focus and Trade-offs: This MVP will focus on core matching features and essential usability. Non-essential features (e.g. complex user management, elaborate analytics, etc.) are out of scope for now. The priority is to get the matching algorithm and flows working end-to-end. The system should be flexible for future enhancement (e.g., adding more data sources or refining the AI model), but those will come in later versions.

By achieving these objectives, the MVP will demonstrate a viable product that can significantly streamline hiring and job search processes through AI-powered matching.

2. Functional Overview

The AI-driven candidate matcher MVP provides two primary user-facing functions (for recruiters and job seekers) and additional supportive functionality. Below is an overview of key features:
	•	Job → Candidate Matching (for Recruiters): Recruiters can input a job description (or select a job posting) and receive a ranked list of candidate profiles that best match the role. For each suggested candidate, the system provides a detailed AI-generated explanation of why this candidate is a good fit (e.g., highlighting matching skills, experience, or keywords from the resume and job description). The matching considers semantic context, not just keyword overlap, using vector similarity to find relevant candidates. For example, if a job requires “Java programming and team leadership,” the system might surface candidates with roles involving software development and project leadership, even if exact keywords differ. Explanations are generated via the LLM to increase recruiter trust and offer quick insights into each recommendation.
	•	Candidate → Job Matching (for Job Seekers): Job seekers can input their resume or profile (or answer a few questions about their experience) and get a list of relevant job postings. These job recommendations are ranked by best fit, and each comes with an AI explanation justifying the match (e.g., “This job matches your experience in data analysis and Python”). The goal is to help candidates discover opportunities that they might miss with keyword search, by using the AI’s understanding of their skills and experiences. This feature can be delivered via a simple web form or an API endpoint where the resume text is provided, and the system returns job titles, company info, and match reasoning.
	•	Interactive Query Agent (Conversational Search): The system supports an interactive, multi-turn query interface (primarily for recruiters) to refine or broaden searches through natural language. A recruiter can perform a multi-step search process, somewhat like chatting with an AI assistant:
	•	Example: The recruiter asks, “Find candidates for a Senior Data Scientist role in New York.” The system returns a few matches. The recruiter then follow-ups with, “Only show those with at least 5 years of experience and a background in finance.” The system understands the context (previous query and results) and refines the list accordingly.
	•	This is enabled by an agent-like flow where the AI keeps track of the conversation state (either implicitly via the conversation history or explicitly via session data) and uses the LLM to interpret follow-up instructions. It uses the vector search to filter or re-rank results based on new criteria.
	•	The agent can clarify queries and handle simple conversational turns. For MVP, the scope of “agent” is limited to query refinement and filtering – complex workflows or tools integration is out of scope. Essentially, it’s a chatbot-style interface on top of the search results, improving the user experience for complex searches.
	•	Performance Monitoring and Logging: Basic monitoring and logging features will be built in (mostly for the administrators/developers of the system, not end-users). Every search query (job-to-candidate or candidate-to-job) will generate logs that include:
	•	Query details (e.g., timestamp, type of query, maybe hashed or anonymized info about input length).
	•	Performance metrics like how long each step took (embedding generation, vector retrieval time, LLM response time).
	•	Outcome metrics such as number of results found, any errors or timeouts from external APIs (like OpenAI).
	•	These logs will be accessible via AWS CloudWatch (since we are using AWS Lambda) or a simple logging dashboard. They help in debugging issues and tracking usage. For MVP, alerting is minimal (perhaps simple alerts for errors or slow response), but the groundwork is laid for more robust monitoring.
	•	Additionally, some basic analytics could be captured (like how often recruiters search for certain skills, or which queries return few results – indicating a gap in candidate data). However, advanced analytics/BI is not a primary focus in MVP.

Production-Ready Considerations: All features above are designed to work in a deployed environment. The matching flows (job→candidates and candidate→jobs) will be exposed via a web UI or API, and the interactive agent may be part of the UI (a chat interface) or accessible via an API endpoint that manages a session. While the MVP will likely have a simple UI, it’s also possible to start with just API endpoints and use a tool like Postman or a minimal web form for testing; the key is that the system is usable and accessible to the intended users (recruiters and job seekers), which in practice likely means a basic web-based interface in addition to the APIs.

3. Technical Stack

To achieve the above functionality with limited development effort and maximum use of existing AI capabilities, the MVP will utilize the following technical stack and components:
	•	Backend Runtime – AWS Lambda (Serverless): All server-side logic will run in AWS Lambda functions, making the solution easily scalable and managed without explicit server provisioning. AWS Lambda is a fitting choice as it can run code without managing servers and automatically scale on demand ￼. This will allow our application to handle variable load (e.g., more queries during business hours) seamlessly. Each core feature might be one or more Lambda functions (for example, one for handling matching queries and one for handling interactive chat follow-ups, or potentially a unified function depending on design). Using Lambda also simplifies deployment (with tools like AWS SAM or Serverless Framework) and reduces ops overhead for the MVP.
	•	API Gateway: The system’s functionality will be exposed via a RESTful API (or GraphQL, but REST is assumed for simplicity in MVP) through AWS API Gateway. The API Gateway will route requests (e.g., a POST request with a job description to /matchCandidates) to the appropriate Lambda, handle authentication (if needed), and enable a secure, public-facing endpoint for clients (including the web UI or external clients). This makes the system truly “hostable” – any recruiter or job seeker app could integrate, and we can easily test via HTTP requests.
	•	Vector Embeddings for Profiles and Jobs: The core matching is based on vector similarity, so we need to represent job postings and candidate resumes in a high-dimensional embedding space. We will use open-source embedding models from Sentence Transformers (SBERT) to generate these embeddings. Sentence Transformers provides many pre-trained models for sentence/text embeddings that capture semantic similarity, which is ideal for our use case (matching descriptions of experiences to descriptions of job requirements). For example, we might use a model like all-MiniLM-L6-v2 for efficiency, or a domain-specific model if available. By using an open-source model, we avoid external API calls for embedding and can fine-tune if necessary. (As a backup or future improvement, we could consider OpenAI’s embedding API for possibly better quality, but for the MVP the focus is on a self-contained solution).
	•	Why Sentence Transformers? Prior work and research indicate that SBERT-based embeddings provide meaningful representations for job recruitment data. They can cluster and categorize job descriptions effectively by meaning, not just keyword matching. In one example, using SBERT for job postings resulted in significant performance improvements over older methods like Word2Vec ￼. This means our system will likely retrieve more relevant matches than a keyword search, capturing nuances like related skills or titles.
	•	Vector Database for Similarity Search: Once we have embeddings, we need to store them in a vector index to perform nearest-neighbor searches efficiently. The plan for MVP is:
	•	During development and local testing, use FAISS (Facebook AI Similarity Search), an open-source library for efficient vector similarity search. FAISS can be integrated into Python easily and perform in-memory searches over thousands to millions of vectors with high speed. It’s well-suited for our initial dataset sizes and allows development without cloud dependencies. FAISS is known for its ability to handle nearest-neighbor search in high-dimensional spaces very quickly (especially with GPU support, though for MVP we can start with CPU).
	•	For production deployment, we will switch to a managed vector database service for scalability and ease of maintenance. The primary option is Pinecone, a cloud-native vector database. Pinecone offers a fully managed service for storing and searching vectors, with virtually no infrastructure work and strong performance at scale. Pinecone is optimized for textual vectors and semantic search use cases, aligning well with our LLM-based matcher. Using Pinecone in production means we can scale to larger datasets (tens of thousands of candidates, for example) without worrying about deploying our own servers for FAISS.
	•	As an alternative (especially for users who prefer open-source stacks or want to keep data within their AWS environment), we consider Amazon OpenSearch (or self-managed OpenSearch) with its k-NN vector search plugin. OpenSearch supports storing embeddings and performing similarity search on them. We could deploy an OpenSearch cluster and use it as our vector store in production if Pinecone is not desirable. This requires more ops work than Pinecone but keeps everything in AWS. The system will be designed abstractly so that the vector index can be switched (FAISS vs Pinecone vs OpenSearch) via configuration.
	•	Index structure: We plan to index both candidates and jobs as vectors. For job-to-candidate matching, we encode all candidate resumes into vectors and build an index. For candidate-to-job, we encode all job postings into a separate index. The indices could be separate, or combined with a field indicating type, but separate indices simplify the logic. Each entry will store the vector plus metadata (like an ID, name/title, maybe some tags for filtering like location or years experience to support the agent queries).
	•	OpenAI LLM for Explanation Generation (and Agent reasoning): For generating the natural language explanations and for interpreting multi-step queries, we will integrate an OpenAI large language model (likely GPT-4 or GPT-3.5 via API). The LLM will not be used to do the actual matching (that’s what embeddings are for) but will be used in a RAG (Retrieval-Augmented Generation) pattern to:
	•	Take the retrieved results (top N candidates or jobs from the vector search) along with the query (job description or resume) as context, and produce a summary or explanation of the match. For instance, given a job description and a candidate’s resume text, the LLM can be prompted to explain why the candidate is a good fit, referencing specific skills or experience from the resume that align with the job requirements. This provides a human-friendly justification for each match, which adds a lot of value beyond a raw similarity score.
	•	Handle interactive agent prompts: the LLM can take a conversation history into account (including perhaps a list of candidates or a summary of them) and a follow-up instruction from the recruiter, and figure out how to modify the search (e.g., filter out those with less experience). In practice, we might implement this by having the Lambda maintain some state (like candidate IDs from the last query) and then the LLM decides which IDs to keep or drop based on the filter request. Alternatively, the LLM could output a new query or new vector search criteria.
	•	We will use OpenAI’s API for this because it’s the fastest way to get high-quality NL output. The specific model might be gpt-3.5-turbo for cost efficiency in MVP, with an option to switch to GPT-4 if the quality difference is significant. The prompts will be carefully designed (few-shot examples of good explanations, instructions to cite specifics from the input, etc.) to ensure we get deterministic and useful outputs.
	•	RAG Rationale: By using retrieval-augmented generation, we give the LLM access to up-to-date and specific data at query time, which dramatically improves relevance and reduces hallucinations. The LLM alone doesn’t “know” our candidate database; but with RAG, it’s provided the relevant snippets (resume sections, etc.) so it can ground its responses in actual data. This also means we can update the candidate/job data independently of the LLM – new entries are immediately considered in searches once their embeddings are indexed, solving the LLM “static knowledge” problem without retraining.
	•	Frontend (Minimal UI): Although the PRD primarily focuses on the backend and core logic, a simple frontend will likely be included for demonstration and usability. The tech stack for the UI can be lightweight (HTML/JavaScript, possibly a simple React app or just plain HTML forms). This UI will interface with the API Gateway endpoints. Key components:
	•	A form for recruiters to paste a job description (or select from preset examples) and a “Find Candidates” button to invoke the job→candidate API. Results (candidates) will be listed with name and the AI-generated explanation.
	•	A form for job seekers to paste a resume (or upload text) and a “Find Jobs” button for the candidate→job API. Results (job titles with company and location, plus explanation) listed.
	•	An interface for the interactive agent: possibly implemented as a chat widget where each user query (initial job search or follow-ups) and the assistant’s answers (list of candidates or confirmations) are displayed. This can be simple for MVP (does not need fancy real-time websockets; a round-trip per message is fine).
	•	We can use a simple Python web framework (Flask or FastAPI) or even just static pages calling the API via JavaScript fetch calls. Since hosting on AWS, an easy way is to use AWS Amplify or an S3 static website for the frontend, or possibly integrate it into the Lambda using API Gateway’s Lambda-Proxy integration for simplicity (not ideal for large scale, but fine for MVP).
	•	Additional Services and Tools:
	•	Data Storage: Candidate resumes and job descriptions can initially be stored in a simple way (for MVP, we may not need a database if we embed everything upfront and just keep data in memory or flat files). However, for a production scenario, storing raw data in a database (SQL or NoSQL) or an object store (S3) is important. We might include an Amazon S3 bucket for storing resumes and job postings (in JSON or text form). The vector index (Pinecone or OpenSearch) will store the embeddings and references to these items. No complex relational schema is needed in MVP; a simple key-value approach (ID -> text) would suffice if needed.
	•	Logging and Monitoring: Use Amazon CloudWatch for logs (Lambda logs automatically go to CloudWatch). We might also use CloudWatch Metrics or X-Ray for tracing if needed. For development convenience, logs can be viewed in the AWS console. We may set up custom metrics for things like “Number of matches returned” or latency, which can be later visualized.
	•	Development Frameworks: Python will be the primary language (given the availability of AI libraries like HuggingFace Transformers, SentenceTransformers, etc., and the ease of writing Lambdas in Python). We may use the LangChain framework for orchestrating the LLM calls and retrieval (LangChain can simplify building RAG pipelines and agent behaviors), although it’s optional. If using LangChain, it could manage the vector store abstraction and the prompting logic for us. Otherwise, custom scripts will do.

Technical Risk & Mitigation:
	•	LLM Hallucination or Inaccuracy: The OpenAI model might occasionally produce an incorrect explanation. Mitigation: use RAG to give it real data (resumes, job text) as context, and instruct it to base the explanation on that data. This is expected to keep answers grounded.
	•	Performance: Embedding a large number of profiles and querying might be slow. Mitigation: use efficient embeddings (small model) and vector indexes. FAISS and Pinecone are built for speed at scale. For example, Pinecone can search millions of vectors in under a second with proper indexing.
	•	Costs: OpenAI API calls and Pinecone have costs. Mitigation: for MVP, limit the number of results and size of context to what’s needed, possibly use cheaper model (GPT-3.5) and moderate dataset size. Also, since Lambda is pay-per-use, cost is proportional to usage; this is acceptable for MVP scale.
	•	Integration Complexity: Using many components (Lambda, Pinecone, OpenAI, etc.) could complicate development. Mitigation: clearly define interfaces between components (e.g., a function that given a query returns embedding; another that given embedding returns top matches; another that given matches returns explanation) so they can be developed and tested independently.

4. Development Timeline & Milestones

To ensure a structured implementation of the MVP, development will be broken into phases with clear milestones and deliverables for each. Below is a proposed timeline with milestones:

Phase 1: Data Preparation and Embedding Index (Week 1-2)
	•	Deliverables: A basic pipeline to ingest sample data (a set of candidate resumes and job descriptions) and create the vector embeddings and indexes.
	•	Tasks:
	•	Collect or create sample data (e.g., 50 sample resumes and 20 job postings in JSON or CSV format).
	•	Write a script or Lambda function to generate embeddings for all items using SentenceTransformer model.
	•	Build an index of candidate vectors (using FAISS for now). Ensure we can query this index with a sample job description vector and get similar candidates.
	•	(Similarly, build an index for job postings to test candidate→job search.)
	•	Milestone Criteria: We have a functioning similarity search offline. For example, given a test job description, the script returns a list of candidate names ranked by cosine similarity. This is verifiable by checking if the top candidates indeed have skills related to the job (manual check). No UI or LLM yet, but the core data and vector logic works.
	•	Notes: Focus on correctness of embedding and retrieval. We might measure the embedding generation time and search latency on this small scale to have a baseline.

Phase 2: Retrieval-Augmented Generation (RAG) Pipeline (Week 3)
	•	Deliverables: Integration of the OpenAI LLM to generate explanations for matches. A function that, given a query, performs vector search and then calls the LLM with the results to produce an output.
	•	Tasks:
	•	Develop a prompting strategy for the LLM. For job→candidate: prompt template might include the job description and a brief summary of each top candidate (like name, top 3 skills) and ask the model to pick the best or explain matches. We might experiment to see if the model should directly rank them or just justify each.
	•	Implement a Python function that takes a job description text, uses the embedding index to find top 5 candidate vectors, retrieves their full text profiles, and then calls the OpenAI API with a prompt containing those profiles and the job description. The output should be an explanation or a list of explanations. For now, format the result as needed (e.g., a JSON with candidates and explanation).
	•	Similarly, implement for candidate→job (this might reuse most of the same code, just swapping which index to query and what context to send to LLM).
	•	Test the output with a few examples. Tune the prompt if the explanations are not specific or accurate enough. Possibly use few-shot by providing an example of a job, a resume, and a good explanation in the prompt.
	•	Milestone Criteria: End-to-end matching logic works in a single call (likely a script or Lambda invocation). For a given input (job or resume), the system returns a list of matches with sensible explanations. Example success: Input a “Data Analyst” job description, output includes a candidate “Alice – has 6 years of data analysis with Python, which matches the job’s Python and data visualization requirements” (an explanation along those lines, generated by the AI from Alice’s resume and job data). The explanations should reference actual data (we can verify that if Alice’s resume indeed mentions Python, the explanation says so – meaning RAG worked).
	•	Notes: This phase might be done in a local environment first. We should keep track of token usage of the LLM to ensure prompts are within limits (e.g., if resumes are long, we may truncate or summarize them before sending to the LLM, or limit to top 3 candidates for explanation to fit in context).

Phase 3: API Development and Basic UI (Week 4)
	•	Deliverables: Deployment of the core logic behind a REST API (using AWS Lambda and API Gateway), and a minimal web UI for interactive access.
	•	Tasks:
	•	Create AWS Lambda functions for the two main endpoints: POST /matchCandidates (takes a job description, returns candidates + explanations) and POST /matchJobs (takes a resume, returns jobs + explanations). The code will utilize the Phase 2 pipeline (embedding search + LLM). Use environment variables or config to choose between FAISS (dev) or Pinecone (prod).
	•	Set up AWS API Gateway with these routes, integrate with Lambda. Configure permissions (Lambda needs internet access to call OpenAI; if using Pinecone, possibly VPC or just API key if Pinecone cloud).
	•	Implement a simple static webpage or small web app that can call these APIs. Likely a single-page app with two tabs (one for recruiter, one for job seeker). The user inputs text and sees results. This can be as simple as an HTML form that calls the API and displays JSON, but for user-friendliness we’ll format the results nicely.
	•	Test end-to-end in a deployed scenario (perhaps using a small AWS test environment). This includes testing CORS if the web app is separate from the API domain.
	•	Basic error handling: Ensure the API returns proper HTTP error codes and messages for issues (e.g., if OpenAI API fails or times out, catch and return a 500 with error message; if input is missing, return 400).
	•	Milestone Criteria: User-facing MVP ready. We (the team) can visit a URL for the web UI, input a job description, and see actual candidate results with explanations coming from the live system on AWS. Similarly for resume to jobs. The interactive agent (multi-step) might not be fully implemented yet in this phase, but the single-shot queries work live. We can say the product is “demoable” at this point to stakeholders.
	•	Notes: Deployment might initially be to a test AWS account. We should document the deployment steps (possibly using a script or CloudFormation template). Also, ensure secrets like OpenAI API keys are stored securely (e.g., in AWS Secrets Manager or at least as encrypted env vars).

Phase 4: Interactive Agent Flow (Week 5)
	•	Deliverables: Enhanced search experience with multi-turn query refinement capability.
	•	Tasks:
	•	Design how to maintain conversational state. One approach: assign a session ID to each conversation (each session ties to a recruiter’s query process). Use an in-memory store or DynamoDB table to store context per session (like the last query embedding or the list of last results).
	•	Implement a new endpoint POST /refineSearch (or reuse the same /matchCandidates endpoint but allow an optional session ID and follow-up query). The Lambda handling this will need to: retrieve prior context based on session, combine it with the new query instruction, and then either filter the previous results or run a new vector search with constraints. Possibly use the LLM to interpret the follow-up: e.g., send it something like “Previous results: [list of candidate profiles or summary]. User says: ‘only those with 5+ years experience’. Question: which of these candidates meet the criteria?” The LLM could answer with a list of candidate IDs that fit, or we parse its answer.
	•	Alternatively, implement rule-based filtering for certain common refinements (like years of experience, location) by extracting those from resumes, so the system can handle it without an LLM for more deterministic results.
	•	Update the UI to support the conversational mode: e.g., after getting initial results, allow the user (recruiter) to type a follow-up in a chat interface. Show the refined results or a message like “Filtered to 3 candidates with >5 years experience.”
	•	Test a full conversation scenario for a few use cases.
	•	Milestone Criteria: Interactive search operational. A recruiter can perform at least one refinement step and get a refined result. For instance, start with a broad search, then narrow down by a criterion, and see that the results update correctly according to the instruction. This should feel like a coherent experience (the user doesn’t have to re-enter the whole query; they just provide the additional constraint). The system should handle at least basic refinements reliably.
	•	Notes: This is a stretch goal for MVP if time allows – if it proves too complex, we might cut down the conversational ability (maybe only allow one refinement or use buttons for filters instead of full natural language). The architecture should, however, be built with future conversation/agent capabilities in mind.

Phase 5: Deployment, Testing & Monitoring Setup (Week 6)
	•	Deliverables: The system deployed in a production-like environment (could be AWS prod account or a robust staging), with monitoring in place and documentation completed. Also, completion of all testing (unit/integration/UAT).
	•	Tasks:
	•	Finalize infrastructure: ensure Pinecone (if used) is set up and loaded with data in the production environment, or OpenSearch cluster is running and indexed. Ensure all environment variables (API keys, Pinecone index names, etc.) are correctly configured.
	•	Conduct load testing for a basic scenario (maybe with a tool like JMeter or Locust) to simulate, say, 10 concurrent users making requests, and see if Lambda concurrency or any component is a bottleneck. With serverless, we expect it to scale, but the OpenAI API has rate limits – note any constraints (perhaps we need to queue or limit requests if too many).
	•	Set up CloudWatch alarms for errors (e.g., any Lambda error invokes an alert email to the dev team).
	•	Write a Deployment Guide and User Guide (as part of documentation) so that others can deploy or use the system. This includes how to update the data (e.g., how to add new resumes or jobs – right now it might require re-running the embedding job).
	•	Complete User Acceptance Testing (UAT) with a few sample end-users (could be colleagues acting as recruiters and job seekers). Gather feedback on result quality and usability. Make small adjustments if needed (for example, maybe the explanation format needs tweaking, or the UI could be improved for clarity).
	•	Milestone Criteria: MVP Launch Ready. All critical features are implemented and tested, the system is stable on AWS, and we have confidence it meets the core use cases. The team can decide to freeze development here and focus on any out-of-scope items for future or just polish.
	•	Notes: This timeline is aggressive (6 weeks total). If any phase goes over time, we might deprioritize the interactive agent (Phase 4) since the core matching (Phases 1-3) is higher priority for MVP. Additionally, the timeline can be adjusted based on team size; tasks can run in parallel (e.g., one developer works on the LLM integration while another sets up the API Gateway, etc.).

5. Epics & User Stories

To break down the work, we outline several Epics corresponding to the major components of the system. Under each Epic, there are User Stories (or developer tasks) with descriptions, acceptance criteria, test cases, and implementation notes. These user stories are written mostly from the perspective of an end-user or admin, to clarify the value, but also contain technical details for implementation.

Epic 1: Data Ingestion & Vector Embedding Setup

Description: Set up the pipeline to ingest candidate and job data and convert them into vector embeddings that can be used for similarity search. This forms the backbone of the matching system – without high-quality embeddings and an index, the AI matching cannot work. This epic covers choosing the embedding model, writing the code to generate embeddings, and storing those in a searchable index (initially FAISS).
	•	User Story 1.1: As a developer, I want to load a dataset of candidate resumes and job descriptions and transform each into a numerical vector representation, so that I can perform similarity searches between jobs and candidates.
	•	Acceptance Criteria:
	1.	A script or service exists that can read raw text for a candidate profile or a job posting and produce a fixed-size embedding vector (e.g., 384-dimensional if using MiniLM) representing the semantic content of that text.
	2.	The embedding generation uses a well-defined model (e.g., sentence-transformers/all-MiniLM-L6-v2 or similar), and the model choice is documented and installed in the environment.
	3.	The system is capable of processing at least 100 documents (resumes or jobs) in one run and outputting their embeddings within a reasonable time (e.g., a few minutes). This ensures it can handle batch updates.
	4.	All embeddings are stored in a vector index structure that supports nearest-neighbor search. For this story, using FAISS in-memory index is acceptable. It should support adding new vectors and querying for the top N most similar vectors to a given query vector.
	5.	The developer can run a test query: provide a sample job description, retrieve top 5 candidate resumes by similarity, and see that the results include expected candidates (qualitative check).
	•	Test Cases:
	•	Functional Test: Given two identical texts (one as “job” and one as “resume”), the similarity search should rank them very highly (because the content is the same, vector distance should be small). If the embedding is working, the similarity score for identical or very similar text should be above a high threshold (e.g., cosine similarity > 0.9).
	•	Diversity Test: Given a job description for “Software Engineer”, ensure that a distinctly different resume like “Sales Manager” is ranked very low in similarity (cosine < 0.2, for instance). This tests that the embeddings capture differences.
	•	Index Test: Insert a known set of vectors (e.g., 10 predefined small strings with known similarities), query the index, and verify the results are in the correct order and relevant. This tests the FAISS index retrieval correctness (we might not do this in detail for MVP, but at least a sanity check using the library).
	•	Performance Test: Time how long it takes to embed, say, 50 documents. Verify it’s not extremely slow (< 1 second per document on average ideally). If using GPU or optimized model, it should be quite fast.
	•	Implementation Notes:
	•	Use the SentenceTransformers library to load the model and encode texts. Ensure the Lambda environment or development environment has the model cached or downloaded (to avoid long cold-start times in Lambda, we might package the model or download at startup and cache it).
	•	Decide on vector size and model: smaller models (MiniLM) are faster but slightly less accurate; larger ones (mpnet or all-MPNet-base) are more accurate but slower. For MVP, lean toward faster model to improve responsiveness.
	•	FAISS: Use a simple index type like IndexFlatIP (if using cosine similarity with normalized vectors) or IndexFlatL2 for Euclidean. Since data size is small, we don’t need IVF or HNSW indexes yet. Also, note that FAISS runs in memory, so on Lambda (which might have limited memory), ensure we only load a reasonably sized index (maybe a few thousand vectors max in MVP).
	•	Plan for switching to Pinecone: design the code so that the vector store is abstracted. Possibly write an interface or use LangChain’s vector store abstraction that can be FAISS or Pinecone based on config. But initially, focus on FAISS to validate logic.
	•	We should also create unique IDs for each resume and job. Perhaps use a simple increment or hash. These IDs will be used to retrieve full text later for LLM input, so store them alongside vectors (FAISS can store just vectors; we may keep a separate list/dictionary mapping index -> ID and metadata).
	•	User Story 1.2: As a system administrator, I want to be able to update the candidate and job indexes with new data so that the system’s knowledge remains current without full redeployment.
	•	Acceptance Criteria:
	1.	There is a documented process or tool to add a new resume (or a batch of resumes) to the index. For MVP, this could be a script that is run manually. It should generate the embedding for the new data and insert it into the vector index (FAISS or Pinecone).
	2.	Similarly, can remove or mark a candidate/job as inactive (since in real systems, you might want to remove filled positions or candidates who got a job). A removal might simply be not handled (MVP can skip deletion if complex) or we plan to rebuild index periodically. But it’s noted.
	3.	In production (with Pinecone/OpenSearch), adding new data does not require re-indexing everything (we can upsert vectors on the fly). Acceptance is that our design supports this, even if not fully automated in the MVP.
	4.	The embedding model used for new data must be the same as initial, to maintain consistency in vector space. This is documented to avoid mistakes.
	•	Test Cases:
	•	After initial index build, run the update process to add a new fake resume. Then search with a job that should match that resume. Verify that before adding, the resume wasn’t in results (obviously), and after adding, it appears in the top results if it’s relevant. This ensures the update process truly integrates new data.
	•	Test adding a job posting and then using a relevant resume to find it in candidate→job search.
	•	(If deletion is implemented) Test removing a resume: after removal, queries that previously returned that resume should no longer do so.
	•	Implementation Notes:
	•	For FAISS, adding new vectors means either keeping the index in memory and adding to it, or rebuilding. FAISS does have add functions for some index types. For MVP, since data volume is small, we might simply rebuild the index from scratch when data changes (which is acceptable in early stage).
	•	Pinecone provides an upsert API which is straightforward. We should design the code to easily switch to using pinecone.upsert calls when on production.
	•	We might not implement a full admin UI for this. It could be developer-operated for MVP (like running a Python script with new data). In a future iteration, an ingestion pipeline and database could be built.

Epic 2: Matching API Development (Job→Candidate & Candidate→Job)

Description: Develop the backend API endpoints that implement the core matching logic. This includes taking an input (job description or resume), performing the vector search, and returning results with explanations. It covers both directions of matching and ensures the results are formatted for consumption by the frontend or other clients.
	•	User Story 2.1: As a recruiter (API client), I want to send a job description to the system and receive a list of top matching candidates with explanations so that I can quickly identify who to contact for the job.
	•	Acceptance Criteria:
	1.	API Endpoint: A POST endpoint /matchCandidates is available. It accepts a request containing a job description (and optionally some parameters like number of results to return, etc.). The request format and endpoint are documented (e.g., JSON body: {"job_description": "...", "num_results": 5}).
	2.	Response Format: The response is a JSON object containing an array of candidate results. Each candidate result includes at least: candidate ID or name, a relevance score or rank, and an explanation string. Example:

{
  "results": [
     {
       "candidate_id": "CAND123",
       "name": "Alice Smith",
       "score": 0.95,
       "explanation": "Alice has 6 years in data science, matching the job’s requirement for 5+ years. She also has strong Python and machine learning experience, which aligns with the role."
     },
     {... more candidates ...}
  ]
}

The exact fields can be adjusted, but must include the explanation.

	3.	Ranking: The candidates are ordered by relevance (most relevant first). This typically correlates with the vector similarity score.
	4.	Explanation Quality: The explanation for each candidate references specific attributes from the candidate that match the job. (We will consider this criterion satisfied if in testing we see that the explanations mention key skills/requirements from the job posting in context of the candidate profile).
	5.	Performance: The API should handle a request within a reasonable time (< ~5 seconds ideally for top 5 matches). This includes the LLM call. If the LLM call takes too long, it should timeout or degrade gracefully (maybe return results without detailed explanations if needed, though for MVP we assume it succeeds).
	6.	Error Handling: If the job description is missing or empty, the API returns a 400 Bad Request with a helpful message. If an internal error occurs (OpenAI down, etc.), return 500 with a generic error message (and log the details).

	•	Test Cases:
	•	Basic Query: POST a sample job description JSON. Verify the response status 200 and that results array is not empty (assuming we have at least one candidate in the system). Check that each result has an explanation that is not just boilerplate but seems relevant. (This is somewhat subjective, but we can have a baseline expectation.)
	•	No Candidates Case: If the database is empty or the job is very unique such that no candidate has a similarity above a low threshold, the system should handle it. Possibly return an empty results array or a message like “No suitable candidates found.” Test by either clearing the index or using a job query that is intentionally mismatched to our test data. Acceptance is that it doesn’t crash and returns a valid response (empty list or graceful message).
	•	Malformed Request: Omit the job_description field or send wrong content type. Expect a 400 with an error message. For example, sending an empty JSON {} should result in a clear error about missing input.
	•	Performance Test: Measure response time with a typical input. If possible, simulate 5 concurrent requests and ensure each still returns within a reasonable range (Lambda should scale, but OpenAI could be a bottleneck; in tests, maybe stub the LLM or use a smaller model).
	•	Content Validation: Check that the explanation indeed corresponds to candidate data. E.g., if we know candidate X doesn’t have a certain skill, the explanation shouldn’t mention that skill. This ensures the LLM is not hallucinating outside the provided context.
	•	Implementation Notes:
	•	The Lambda for /matchCandidates will do: parse input -> embed the job description -> vector search on candidates index -> take top K results -> assemble context (e.g., for each candidate, perhaps use their name, title, and a summary of experience) -> call OpenAI LLM with a prompt to generate explanation text for each (or a combined explanation).
	•	One approach: call the LLM once per candidate to get a personalized explanation. That could be slow (5 calls). Instead, we might craft one prompt that lists all top candidates with brief details and ask the LLM to write a short blurb for each. We must be mindful of token limits. For MVP, doing multiple calls might be acceptable if K is small.
	•	Caching: We might implement a basic cache for embeddings or LLM calls (perhaps not needed in MVP, but e.g., if the same job description is queried multiple times, we could reuse results).
	•	If using Pinecone in prod: ensure network calls to Pinecone are efficient (the latency should be low, Pinecone is quite fast < 100ms typically).
	•	The explanation prompt might look like:

You are an assistant helping match job descriptions to candidate resumes. 
The job description is: "[JOB TEXT]". 
Candidate: [Name] – [some key skills or summary]. 
Explain why this candidate is a good fit for the job.

We might include multiple candidates in one go by repeating candidate sections and asking the model to output a numbered list of explanations.

	•	We have to include only relevant parts of the resume in the prompt, to avoid using too many tokens and to focus the model. Possibly pre-extract each candidate’s top skills or most recent experience for the prompt.
	•	Logging: log the request (without full text maybe, to avoid PII in logs, or at least truncated) and log the outcome (how many results, etc.).

	•	User Story 2.2: As a job seeker (API client), I want to send my resume or profile to the system and receive a list of matching job postings with explanations so that I can identify which jobs to apply for.
	•	Acceptance Criteria:
	1.	API Endpoint: A POST endpoint /matchJobs exists, accepting a resume or candidate profile text (or structured data) and optional parameters (like number of results).
	2.	Response Format: JSON response containing an array of job postings. Each job result includes: job ID or title, company (if known), maybe location, a relevance score, and an explanation of the match. Example entry:

{
  "job_id": "JOB456",
  "title": "Senior Data Analyst",
  "company": "TechCorp Inc.",
  "score": 0.88,
  "explanation": "Matches your experience in data analytics and reporting. The job asks for SQL and Tableau, which you have used extensively."
}

Fields like company and location would come from the job data if available.

	3.	Ranking: Jobs are ordered by relevance to the candidate’s profile.
	4.	Explanation: Should highlight key qualifications the candidate has that the job requires, or aspects of the candidate’s background that align with the job role.
	5.	Behavior: Similar error handling and performance requirements as the previous story (within 5 seconds, handle empty input, etc.).
	6.	The system should not return jobs that are clearly not a fit (this is subjective, but basically trust the similarity; if the resume is for marketing, it shouldn’t return engineering jobs at top ranks).

	•	Test Cases:
	•	Provide a sample resume that has a known target job in the data. Verify that job is among the top results. For example, if resume mentions a lot of finance experience, and we have a “Financial Analyst” job posting, that should show up with a high score.
	•	Provide a very general resume (e.g., a student with minimal info) and see that the system still returns something reasonable (maybe entry-level jobs or an empty result if nothing is appropriate).
	•	Malformed request tests similar to 2.1 (missing resume field, etc.).
	•	If possible, test with a resume in which certain skills are phrased differently than in jobs (e.g., resume says “created web dashboards”, job says “experience with web analytics tools”). The embedding should catch this semantic link. The explanation hopefully bridges that (“your experience creating web dashboards aligns with the requirement for web analytics”).
	•	Ensure that any sensitive personal info in resume (if present) is not just parroted blindly in the explanation. The explanation should focus on professional match aspects. (This is more of an AI prompt quality test).
	•	Implementation Notes:
	•	This will share a lot of logic with the /matchCandidates implementation. We may generalize a function that given an input text and a target index (candidates or jobs) returns the matches and explanations.
	•	Pay attention to prompt differences: for a resume → job, the prompt should be phrased accordingly (e.g., “The candidate has these experiences: [summary]. The job is [job description]. Explain why the job is a good fit for the candidate.”).
	•	Since job postings might be long (full description), we might need to truncate or summarize them for the prompt as well. Possibly only include the key requirements section or first paragraph of the job description in the LLM context, to save tokens.
	•	We might also store a short summary of each job along with its vector (precomputed), to use in prompts instead of full text.
	•	Data format: If resumes are uploaded as PDF or DOCX, an OCR/text extraction step is needed out of scope for MVP – we assume text input. We could note this as a limitation that resumes must be provided in text form for now.

Epic 3: LLM Explanation & Agent Flow

Description: Focus on the AI agent capabilities – generating the explanations and handling multi-step conversational queries. This epic ensures the system’s AI responses are coherent, useful, and that the multi-turn interaction is functional.
	•	User Story 3.1: As a recruiter, I want the system to explain why each recommended candidate is suitable for my job, so that I have confidence in the recommendations and can communicate the reasoning to my hiring team.
	•	Acceptance Criteria:
	1.	For any job search result (from story 2.1), an explanation is provided for each candidate. The explanation should mention at least one specific skill, qualification, or experience that the candidate has and that is relevant to the job description.
	2.	The explanation should be in natural language, one to three sentences long, and should read like a brief note from a recruiting assistant. It should not contain raw fragments of text that seem out of context (i.e., it should be coherent and not just copy-pasted bullet points).
	3.	The explanation must not introduce information that is not present in either the job description or the candidate’s profile (to avoid hallucination). In other words, it shouldn’t say “Candidate is a perfect fit” without backing, or add skills that neither the job nor resume mentioned.
	4.	If the matching is weak (e.g., no great candidates), the explanation might say something like “This candidate has some related experience in X which partially matches the role requirements,” indicating a partial match rather than a strong endorsement. (This is more advanced; for MVP, even stating “Candidate has X and Y required by the job” is fine.)
	5.	The language model’s output is sanitized: no offensive or inappropriate content (shouldn’t be an issue since input is professional text, but we ensure the prompt asks for a professional tone).
	•	Test Cases:
	•	Check a few output examples manually to ensure they meet criteria. For example, if a job asks for “MBA preferred” and the candidate doesn’t have an MBA, the explanation shouldn’t claim the candidate has an MBA.
	•	We can design a controlled test: Feed a known small context to the LLM (like job wants skill A and B; candidate has A and C). See if the explanation mentions A (good) and hopefully not mention B as candidate having it. If the LLM output is incorrect, adjust prompt or approach (maybe list explicitly what skills the candidate has to constrain it).
	•	Test multiple domains (IT job, marketing job, etc.) to ensure the model handles various jargon and still makes sense.
	•	Ensure the explanation is concise. If the LLM tends to write very long paragraphs, adjust instructions to limit length.
	•	Implementation Notes:
	•	This is largely about prompt engineering. We might include instructions like “Be concise. Focus on relevant skills and experiences. Do not add information that is not given.”
	•	Possibly use few-shot examples: e.g., provide one example in the prompt: “Job: … Candidate: … Explanation: …”, to guide style.
	•	The OpenAI API should be called with temperature parameter low (e.g., 0.2) for deterministic output, since we want consistent quality and not too much creativity.
	•	We will need to pass the candidate’s details to the model. We should decide how to format that (maybe a bullet list of candidate’s key points vs a raw resume text). Summarizing resume into key points (education, skills, experience summary) prior to feeding might improve the explanation relevance.
	•	If an explanation for each of say 5 candidates is too long to do in one pass, we might loop the model call per candidate with the job description context. This parallel or sequential calls could slow things down but ensures focus. We can see performance trade-offs.
	•	User Story 3.2: As a recruiter, I want to refine my search through a conversational interface, so that I can iteratively improve the candidate results (e.g., narrowing down by experience or location) without starting over.
	•	Acceptance Criteria:
	1.	After performing an initial search (e.g., via /matchCandidates with a job description), the system supports a follow-up query that refines those results. This can be via a new endpoint (like /refineCandidates) or by including a conversation ID and new query in the same endpoint.
	2.	The follow-up query can be phrased in natural language. E.g., the user can input: “Only show me candidates in New York with 5+ years of experience.” The system will interpret this in context of the previous results or search criteria.
	3.	The output will be a new filtered/re-ranked list of candidates that meet the additional criteria. If none of the previous candidates meet it but others in the database do, the system might consider fetching those. (MVP might keep it simple and only filter the already retrieved top N, which is easier.)
	4.	If the request is something that cannot be applied (e.g., “show those with certification X” and we don’t have that data), the system should either respond with a clarification or handle it gracefully (maybe ignoring that filter or saying “No data on X available”).
	5.	The conversation state is maintained so that the user doesn’t need to repeat the original job description. The system implicitly knows what pool of candidates or what job context we are talking about after the first turn.
	6.	There should be a limit to the number of turns (for MVP, even 1 follow-up might be enough, or maybe up to 3 turns) to avoid indefinite conversation management complexity.
	•	Test Cases:
	•	Perform a search for a broad job (many results), then do one refinement by location. Verify that all returned candidates indeed have that location in their profile (we should have location data for candidates to test this; if not, choose another filter).
	•	Try a refinement that our data can’t support, like “with at least 10 publications” when we don’t have info on publications. The expected behavior might be that the system just ignores it or says “Sorry, I cannot filter by publications.” Test for a graceful message or that it doesn’t break.
	•	Ensure that if the user does a follow-up that broadens criteria (less likely, but e.g., “actually include those with 3 years experience too”), the system can handle it (this might just re-run original search).
	•	Test session isolation: if two conversations are going on (two different users), their contexts don’t bleed into each other. This is more a concurrency test; ensure the context is tied to a session ID.
	•	Implementation Notes:
	•	Could use a simple mechanism: when initial search happens, store the results or the query embedding and results in a temporary store (in memory dict or DynamoDB) keyed by a session token. Return the session token to the client.
	•	On refine, client sends session token with the query. The Lambda then retrieves context. Alternatively, use JWT or some state in client to send back the whole previous query info (but that’s not ideal for user).
	•	Natural language understanding of refinement: We might utilize the LLM here. Example approach: have the LLM analyze the follow-up text to produce a structured filter (like field = location, value = New York; field = experience, value = >=5). If our candidate data is structured (if resumes parsed into structured fields), we can then apply filters. If not structured, we may have to re-embed the query combined with original description.
	•	Simpler approach: Recognize a few keywords in follow-up (like years, experience, location names) with regex or a small NLP. MVP can hard-code a few for demo purposes.
	•	Time constraints: If this is proving too complex by Phase 4, a simplified version is to allow a second query that just appends to the original text. E.g., user says “with 5+ years experience”, we modify the original job description text to add “must have 5+ years experience” and re-run the whole search. This might actually work well with the embedding approach, as that additional criterion will change the query embedding and thus change which candidates are similar. We can do this as a trick: just treat the follow-up as an addition to the original query and do a fresh vector search. This avoids needing to filter the precise earlier list.
	•	We will clarify to the user (in UI) that they can refine by adding criteria. Perhaps the UI will capture that and call our refine endpoint appropriately.
	•	User Story 3.3: As the system, I want to log the interactions of the conversational agent so that we can review how users are refining their searches and improve the agent’s understanding over time.
	•	Acceptance Criteria:
	1.	Every query (initial or follow-up) handled by the agent is logged with the text of the query, any interpreted filters or actions, and the outcome (how many results, any errors).
	2.	The logs for conversation flows should include a conversation/session identifier to link turns. This will allow debugging a whole session if something went wrong in the logic.
	3.	These logs should be easily retrievable by developers (likely in CloudWatch or a log file). They may be more detailed than the general system logs because this is a complex feature; possibly include the prompt sent to the LLM for refinements, etc., to diagnose if the LLM misunderstood something.
	4.	No sensitive candidate or job data should be logged beyond maybe IDs or short descriptors, to maintain privacy (the raw resumes need not be fully logged).
	•	Test Cases:
	•	Initiate a conversation with one refinement. Then go to the AWS CloudWatch logs and verify that there are log entries that show something like:
	•	“Session 1234: Received initial query for Job X, returned 5 candidates.”
	•	“Session 1234: Received refinement ‘only NY’, applied filter location=NY, returned 2 candidates.”
	•	Check that the logs do not contain the entire resume text or job description, but maybe truncated or just an ID reference. (We want to avoid huge logs and privacy issues).
	•	Force an error (maybe simulate OpenAI failure in refinement) and see if the logs capture that event with enough detail to troubleshoot.
	•	Implementation Notes:
	•	Logging here can be simply using Python’s logging library or print statements (which go to CloudWatch via Lambda). We might structure them in JSON for easier querying if needed.
	•	Because logs are important for an agent, we might set a specific logger for the “Agent” component.
	•	These logs could later be analyzed to improve the system (like if many users ask for a certain filter we don’t support, we know to add it).

Epic 4: User Interface & UX (Recruiter and Candidate Portal)

Description: Provide a minimal user interface that allows recruiters and job seekers to use the system without technical knowledge. Though the core is API-driven, a simple web UI will greatly enhance the demonstration and usability of the MVP. This epic covers designing and implementing that UI.
	•	User Story 4.1: As a recruiter, I want a simple web page where I can paste a job description and click a button to find candidates, so that I can use the matcher without needing to call APIs manually.
	•	Acceptance Criteria:
	1.	A webpage (accessible via a browser) is available with a form for job input. It should have a large text box for the job description, and a submit button labeled appropriately (e.g., “Find Candidates”).
	2.	Upon submission, the page shows a loading indicator (so the user knows the search is in progress, since it might take a few seconds).
	3.	The results are displayed on the page in a readable format: list each candidate with a heading (name or identifier) and the explanation text. Optionally include a score or percentage match.
	4.	If there is an error (no results or system issue), the page displays a user-friendly message (“No candidates found for the given description” or “There was an error processing your request, please try again”).
	5.	The page design should be clean and simple, no need for fancy graphics. Company logo or name of the tool at top is nice to have. The layout should be responsive enough or at least not break on different screen sizes.
	•	Test Cases:
	•	Manually test by opening the page in a browser, inputting a known job description (maybe from our test data). Verify the results shown match what the API returns (we might cross-check the network call).
	•	Test with an empty text box and clicking submit – the page should prompt for input instead of calling the API with empty data (could use HTML5 required field or manual check).
	•	Cause an error (perhaps shut off the API temporarily) and see that the error message is shown gracefully.
	•	UI sanity: test on a mobile screen dimension to see if text is still readable (not a major focus, but just that it’s not completely broken).
	•	Implementation Notes:
	•	The UI can be a single HTML file with some JavaScript. We can host it on S3 as a static site or use AWS Amplify for simplicity.
	•	Use fetch in JS to call the API Gateway endpoint. Because of cross-origin, ensure CORS is enabled on API Gateway for our domain or ‘*’ for simplicity during MVP.
	•	Could use a simple CSS framework like Bootstrap or just minimal custom CSS for readability.
	•	Think about how to display the explanation: maybe italicize it or put it in a quote block under each candidate name for clarity.
	•	Also consider linking to candidate full profile or job full description if this info is available (maybe not in MVP).
	•	User Story 4.2: As a job seeker, I want a similar interface where I can input my resume text and see matching jobs, so that I can easily find opportunities.
	•	Acceptance Criteria:
	1.	On the same site or a separate page, provide a form for resume input (multi-line text area). Alternatively, allow file upload (stretch goal, if we have client-side file-to-text conversion).
	2.	After clicking “Find Jobs” or similar, show a list of jobs with titles and explanations as to why they fit the user.
	3.	If possible, show key details of the job (title, company, location, snippet of description).
	4.	Must handle large text input (resumes can be long). The form should accept a decent amount of text (if using HTML <textarea>, ensure no extremely small character limit).
	5.	Similar error and loading states as in 4.1.
	•	Test Cases:
	•	Input the sample resume corresponding to known jobs and verify the UI list matches expectation.
	•	Try partial resume (just a few skills) to see how the system behaves and that the UI still shows something reasonable.
	•	Ensure that if the text is very long and exceeds perhaps URL length (though we use POST, so it’s fine), the system still works. Possibly test a truncated scenario.
	•	Implementation Notes:
	•	Likely reuse the same script logic as the recruiter side, just calling the other endpoint and rendering results slightly differently (job title bold, etc.).
	•	We might have a single page with two tabs: “For Recruiters” and “For Candidates,” switching which form is visible. This is convenient for one static site.
	•	Keep in mind that the resume or job description text is sensitive; ensure the page is served over HTTPS (if using API Gateway domain it is by default). Also perhaps include a note like “No data is stored, it’s only used to fetch results live,” to reassure users (if that’s true).
	•	User Story 4.3: As a user (recruiter or job seeker), I want the interface to allow me to refine my query in a conversational way, so that I can get to the right results through follow-up instructions (if this feature is implemented).
	•	Acceptance Criteria:
	1.	After initial results are shown, the UI provides a way to enter a follow-up query or instruction. This could be a chat-like interface or just an additional text input that appears with a label like “Refine your search:”.
	2.	When the user submits a refinement, the UI either updates the list in place or shows a new list, and possibly records the conversation (e.g., showing the previous query and new filter somewhere for context).
	3.	If the conversation is more than one turn, the UI should show a history (like the original query and the refinement). For MVP, even showing just the latest result might be okay, but ideally we indicate what filters have been applied.
	4.	The user can reset the search easily (a “Start over” button) to do a completely new search if needed.
	5.	The refine interface should handle the same error states (if refine returns nothing, maybe tell “No candidates match that refinement”).
	•	Test Cases:
	•	After getting results for a broad query, enter a refinement and ensure new results show. Check that they are indeed refined (if filter by location, all displayed have that location).
	•	Try multiple refinements sequentially (if allowed) and see if the system keeps narrowing down (within the UI).
	•	Click “Start over” and verify it clears the conversation and allows a new initial search.
	•	Implementation Notes:
	•	This requires the UI to manage some state (the session ID from the backend or simply the last query data). Possibly we handle it by keeping the last results in JS and filtering, but since our backend might do better, we will call backend with refine.
	•	If using a chat UI approach, we might append the user’s request and system response to a chat log div.
	•	Simpler: show a text input “Refine results:” and on submit, call the refine API with session ID, then just replace the results list with new ones. Perhaps also show a note “Filtered by: [user’s refine text]” for context.
	•	This part is a bit more complex for UI but can be rudimentary. It might not be fully realized if time is short.

(If UI development is not feasible due to time, the API itself suffices for MVP functionality, but a basic UI is highly preferred to fulfill the usability objective.)

Epic 5: Monitoring, Logging & Performance

Description: Implement the necessary logging, monitoring, and performance optimizations to ensure the MVP runs reliably in production and can be maintained.
	•	User Story 5.1: As a system admin, I want the system to log key events and errors, so that I can troubleshoot issues and understand system usage.
	•	Acceptance Criteria:
	1.	Every API call (matchCandidates, matchJobs, refine, etc.) results in a log entry with at least: a timestamp, the operation name, and outcome (success/failure). For successes, log the number of results returned and time taken. For failures, log an error message and stack trace if applicable.
	2.	Sensitive data (full text of resumes/jobs) is not logged in plain text. Instead, log metadata like lengths, IDs, or a truncated snippet if needed.
	3.	A monitoring dashboard or at least CloudWatch metrics are set up to track: number of requests, average latency, number of errors. Could be as simple as printing metrics and relying on CloudWatch Logs Insights, or explicitly putting custom metrics.
	4.	The system should have an easy way to toggle verbose logging (for debugging) vs minimal logging (for production) through configuration.
	•	Test Cases:
	•	Trigger an error (like simulate OpenAI failure by providing a bad API key) and verify that an error log with relevant info appears in CloudWatch.
	•	Make a few requests and then use CloudWatch Insights or Metrics to confirm that counts and latencies were recorded.
	•	Check that no full resume text appears in logs by searching the logs for a unique phrase from a resume.
	•	Implementation Notes:
	•	Logging can be implemented directly in Lambda code using Python logging module. Set log level via an environment variable.
	•	Use structured logging (JSON) if possible for easier querying.
	•	For metrics, we can use cloudwatch.put_metric_data in Lambda or simply rely on default metrics like Lambda duration and count. If detailed metrics needed, maybe incorporate a library or CloudWatch Embedded Metric Format.
	•	Ensure the OpenAI API response and Pinecone responses are also error-handled with try-catch and logged.
	•	User Story 5.2: As a product manager, I want to monitor the quality of matches over time, so that I can ensure the AI is performing well and identify when it needs improvement.
	•	Acceptance Criteria:
	1.	(Stretch) The system should collect some feedback on match quality. Since this is hard to get automatically, for MVP, we might just log the similarity scores and maybe the prompts and outputs for review.
	2.	Perhaps provide an internal endpoint or method to manually review a random sample of matches (for internal testing).
	3.	Define a couple of simple quality metrics: e.g., average similarity score of top result, or coverage (percent of queries that returned at least one result). These can be derived from logs.
	4.	Not necessarily automated in MVP, but there is a plan noted for evaluation (which could be done offline by examining results).
	•	Test Cases:
	•	This is more analytical: run a set of test queries and manually verify if the results seem good. Record those observations. Not a coded test, but acceptance is more subjective here.
	•	Implementation Notes:
	•	We may skip a formal implementation here due to MVP focus, but we’ll ensure logs have enough info to compute these if needed (like log similarity scores and maybe which candidate was chosen as best).
	•	Possibly include a hidden flag in API to output debug info (like the raw scores or the top 10 before LLM filtering) for internal use.
	•	User Story 5.3: As a DevOps engineer, I want the system to be designed for easy scaling and deployment, so that it can handle increased load and be quickly updated.
	•	Acceptance Criteria:
	1.	The Lambda functions are stateless and can scale out (we inherently get this with Lambda, but we ensure no global state issues in code).
	2.	Cold start times are within acceptable range (maybe a couple of seconds at most). Using a smaller model and possibly provisioning concurrency if needed to keep one instance warm could be considered.
	3.	The deployment process (using AWS SAM/Serverless or manual steps) is documented and can be executed in one go (e.g., one command to deploy all).
	4.	The system can handle at least 100 requests per day easily (MVP goal; scaling beyond that likely fine with Lambda, but we want to ensure no obvious bottleneck).
	5.	If the vector DB is Pinecone, ensure that the free tier or chosen tier can handle our vector count and query rate. If OpenSearch, ensure the cluster has appropriate instance sizing (not too low on memory for the vectors).
	•	Test Cases:
	•	Simulate a burst of requests (maybe 10 at once) and observe that Lambdas scale (no throttling, and responses all succeed). Check CloudWatch concurrent executions metric if needed.
	•	Simulate a deployment on a fresh AWS account using our instructions to ensure nothing is missing (like IAM roles, API keys).
	•	If possible, test switching the vector backend from FAISS to Pinecone easily by configuration to confirm our abstraction works.
	•	Implementation Notes:
	•	This is ensured by design: using AWS Lambda and API Gateway already addresses much of scaling.
	•	The main possible scaling issue is the external calls (OpenAI has a rate limit of maybe 20 requests/min for free or certain accounts). That might be a constraint; if expecting more usage, consider request queueing or split into multiple OpenAI API keys.
	•	Another is Pinecone (depending on plan, there’s a limit on queries per second).
	•	Use environment config in Lambda for keys and endpoints so that updating them doesn’t require code change.
	•	We should also handle timeouts: Lambda max runtime by default is 3 seconds? Actually can configure up to 15 min, but we likely set 10 sec as our function timeout to not hang too long. If LLM doesn’t respond by then, function times out—this should be set with care.

6. Testing Approach

A comprehensive testing strategy will be adopted to ensure the MVP meets the requirements and performs reliably:

Unit Testing:
	•	We will write unit tests (using Python’s unittest or pytest) for the core logic functions:
	•	Embedding function: Test that embedding generation returns a vector of expected dimension and type. If using a specific model, test that identical sentences give identical embeddings (deterministic) and different sentences give different embeddings.
	•	Similarity search function: If we wrap FAISS/Pinecone calls in a function, test that when we input a vector that is exactly one of those in the index, it returns that item as top-1 result.
	•	Prompt generation function: If there’s a function that prepares the prompt for the LLM, test it with sample inputs to ensure it includes all necessary parts (job description, candidate info, etc.) and doesn’t exceed length limits. Could also test that it handles special characters properly (no JSON or formatting issues that might confuse the model).
	•	Response parsing function: If we parse the LLM output (for example, if we ask it to output structured text or JSON), test that parsing works for expected formats and handles slight variations robustly.
	•	Utility functions: e.g., anything for filtering refinements (if we parse “5+ years” into a number, test that).
	•	These tests will be run locally. For parts that call external services (OpenAI, Pinecone), we might use mocking. For example, mock the OpenAI API call to return a fixed string (to test our parsing and handling logic without incurring cost or dependency).
	•	Unit tests will also cover error scenarios: simulate exceptions thrown by vector search or by the LLM and ensure our code catches and handles them (returns appropriate error messages).

Integration Testing:
	•	Integration tests will test the end-to-end flow in a controlled environment. We might set up a small dataset and run the whole pipeline:
	•	Deploy the Lambda (or run a local instance using something like serverless invoke or SAM local) with test event payloads. Verify the response end-to-end is correct.
	•	Test the API through API Gateway (could use a tool like Postman or requests in a script) to ensure the HTTP layer is working (CORS, headers, etc.).
	•	Integration test scenarios:
	•	A typical job->candidates query with known expected output (if we control the data, we can assert that candidate X is among results).
	•	A resume->jobs query similarly.
	•	A refine query after an initial query (this might require calling the real deployed endpoints in sequence, maintaining session).
	•	We will test on a staging deployment (perhaps an AWS dev stage) before production. This ensures that IAM roles, network access (e.g., Lambda’s internet access for OpenAI API) are all correct.
	•	Integration tests might not be fully automated in code due to complexity, but a script to exercise the main functions and print results can be used for quick checks.

Performance Testing:
	•	Since this is an MVP, we won’t do heavy load testing, but we will measure:
	•	Average latency of a single request (and see if it’s within our target ~3-5 seconds for a moderate size input).
	•	If possible, test concurrency by firing 5-10 requests simultaneously (maybe using a simple loop or a tool). This is to see if any contention (e.g., model loading in Lambda causing slowness).
	•	Memory usage: ensure the chosen Lambda memory size is enough to hold the model and process (monitor if any out-of-memory or consider raising memory which also increases CPU).
	•	If performance is inadequate, we may need to adjust (e.g., reduce number of results or use a smaller model). This testing will guide such tuning.

User Acceptance Testing (UAT):
	•	Once the system is deployed and stable, we will conduct UAT with a few sample end-users:
	•	For recruiters: provide them access to the UI and have them try to find candidates for a job they know. Collect feedback on whether the results make sense, if the explanations are helpful, and if the interface is easy enough.
	•	For job seekers: similar, have someone input their actual (or sample) resume and see if the job suggestions align with what they’d expect or if there are any surprising matches.
	•	UAT will focus on usability aspects: did the user understand how to use it? did they trust the results? were there any confusing parts?
	•	We will document any issues discovered (for example, maybe the explanation uses too much technical jargon, or the UI is not clear about something) and decide if any quick fixes are needed in MVP or left for next iteration.

Regression Testing:
	•	If we make changes (like update the embedding model or tweak the prompt) during development, we will re-run earlier tests to ensure nothing else broke. Because of the stochastic nature of LLMs, we have to be careful – ideally lock down the LLM behavior with consistent prompts and low temperature so that tests are repeatable to some extent.
	•	We might save some known outputs as reference (not strictly assert equal because LLM text can vary, but assert that certain keywords appear in the explanation, for instance).

Testing Limitations:
	•	It’s hard to automatically test the “quality” of matches. We rely on manual inspection for that. We can however test that certain known relevant/not relevant pairs behave as expected by the similarity metric.
	•	The OpenAI part introduces nondeterminism; we’ll mitigate that with fixed random seeds where possible and consistent configuration.

Bug Tracking:
	•	We will use a simple issue tracker to log bugs found during testing and their resolution, ensuring we don’t forget any edge-case fixes.

By combining unit tests, integration tests, and user testing, we aim to deliver a robust MVP that meets the defined requirements and can be confidently demonstrated or trialed with real users.

7. Hosting & Deployment Plan

The MVP is designed for deployment on AWS using a serverless architecture. Below is the plan for how it will be hosted and deployed:
	•	AWS Account and Permissions: We will use an AWS account (or a specific AWS environment within the company’s account) to host the resources. Ensure that appropriate IAM roles are set up:
	•	A role for Lambda with permissions to CloudWatch Logs (for logging), access to Secrets Manager (to retrieve API keys for OpenAI, etc.), and VPC access if needed (for Pinecone or OpenSearch if in a VPC).
	•	If using OpenSearch on AWS, the Lambda role might need permissions to access that (or we ensure network access is configured properly).
	•	Lambda Functions:
	•	We expect to create 2-3 Lambda functions: e.g., MatchCandidatesFunction, MatchJobsFunction, and possibly RefineSearchFunction (if separated). We might combine them into one function with different routes for simplicity, but separate functions give clearer separation of concerns and can be scaled independently if needed.
	•	Each Lambda will be deployed (likely using a deployment package or container image if size is big due to ML libs). We might leverage Lambda layers for large libraries like sentence-transformers to avoid hitting the size limit.
	•	Memory and timeout settings: Initially, allocate a higher memory (e.g., 1024 MB) to ensure the model can load and the CPU is faster (Lambda gives more CPU with more memory). Timeout maybe 10 seconds (should be enough for a few LLM calls in sequence).
	•	Lambda will use environment variables for configuration (like OPENAI_API_KEY, PINECONE_API_KEY, index names, etc.). This keeps secrets out of code.
	•	Vector Store in AWS:
	•	For development, FAISS runs in-memory inside Lambda. However, Lambda is transient (no long-term memory across invocations). To persist an index between invocations, options include: storing the FAISS index in S3 and loading it each time (could be slow), or reconstructing on cold start (also slow if many vectors). This is a strong reason to use Pinecone in production.
	•	Pinecone (Production): We will create a Pinecone index (dimension = embedding size, say 384, with appropriate metric cosine or dot). Pinecone is a managed service outside AWS, but accessible via API. We’ll store Pinecone API keys in AWS Secrets Manager or env vars. The Lambda will call Pinecone’s REST API to query or upsert vectors. We have to ensure the Pinecone environment is appropriate (maybe use their free tier for MVP, which might allow a limited index size and QPS).
	•	OpenSearch (Alternative): If we opt for AWS OpenSearch, we’d launch an OpenSearch domain (managed by AWS). That requires a few minutes to set up and costs for the instance. OpenSearch k-NN plugin would be enabled. The index and vectors would be stored there. Lambda would query OpenSearch via its REST API as well (so Lambda needs network access to the OpenSearch domain endpoint; likely it would be within the same VPC or open to public if security allows).
	•	Considering time and complexity, Pinecone might be the faster path for MVP, as it avoids managing our own cluster and likely to have better performance out-of-the-box. We’ll keep OpenSearch as a contingency or future plan if needed.
	•	OpenAI API:
	•	The OpenAI LLM is accessed via internet, so ensure the Lambda is in a subnet with internet access (if using VPC, attach a NAT gateway, etc., or simplest: don’t put Lambda in VPC if not needed for Pinecone).
	•	The OpenAI API key will be stored in AWS Secrets Manager. The Lambda at startup will fetch it (or we inject it as an env var via Terraform/CloudFormation, but Secrets Manager is safer).
	•	Rate limits: For initial usage, a single key should suffice. If expecting more usage, we might need to request rate limit increase or multiple keys (or caching responses).
	•	API Gateway:
	•	Configure REST API with resources: /matchCandidates, /matchJobs, /refineSearch (if used). Methods POST on each, integrated with the respective Lambdas.
	•	Enable CORS on these endpoints to allow our web front-end domain to call them.
	•	Stage: maybe use a dev stage and a prod stage on API Gateway, so we can deploy and test on dev stage then promote to prod.
	•	Throttling/quota can be set on API Gateway if needed to prevent abuse (for MVP, not critical unless we open it publicly).
	•	Web Front-End Hosting:
	•	Host the static front-end on Amazon S3 (as a static website) or using AWS Amplify Hosting. The front-end will consist of static files (HTML, CSS, JS).
	•	If using S3 static site, ensure bucket is configured public (or use CloudFront for nicer domain/HTTPS). Since this is internal MVP, even opening the HTML from local disk could work, but better to host it.
	•	Amplify might be overkill, but it does CI/CD for front-end if we anticipate updating UI often. Could skip for MVP.
	•	Another approach: serve the UI via an API Gateway endpoint (using Lambda to return HTML). Simpler to just use S3 though.
	•	Deployment Automation:
	•	Use AWS SAM or the Serverless Framework to define the infrastructure as code. This will include Lambda functions (with code bundled or pointing to a Docker image ECR if needed), API Gateway config, necessary IAM roles, and possibly the OpenSearch domain if we go that route.
	•	Alternatively, use Terraform for infrastructure and separate code deployment. But given time, using Serverless Framework YAML might be quickest.
	•	We will script the build and deploy steps: for example, a developer can run sam build && sam deploy --config-env prod to deploy to prod environment. The PRD expects a plan, so yes we’ll have these steps documented.
	•	Domain and DNS: Not strictly needed for MVP (we can use the default API Gateway URL and an S3 website URL). If this were a more public demo, we might set up a custom domain for the API and UI. For now, assume default endpoints.
	•	Monitoring & Alerts:
	•	Use CloudWatch for monitoring Lambda (invocations, errors, duration). Set up an alarm for if errors > 0 in a 5-minute interval, notify the dev team (via email/SNS).
	•	Use CloudWatch alarms for high latency if needed (maybe if 95th percentile duration > some threshold).
	•	Pinecone has its own usage metrics we can monitor via their dashboard (if using Pinecone).
	•	If OpenSearch is used, monitor cluster health and memory (via AWS OpenSearch console).
	•	Cost Management:
	•	The use of serverless means we only pay per invocation and usage. The main cost will be OpenAI API calls. We’ll keep track of usage to not exceed budget. If needed, we can set quotas or use a smaller model. Pinecone has a cost too but for MVP with small index it might be free or low tier.
	•	It’s worth noting that if usage grows, for a production system we would consider more cost-optimized approaches (like using an open-source LLM to avoid API cost, but that’s beyond MVP).

Deployment Steps Summary:
	1.	Set up AWS environment (IAM roles, secrets for API keys).
	2.	Deploy infrastructure (API Gateway, Lambdas) via IaC template.
	3.	Load initial data: run the embedding script to populate Pinecone or prepare FAISS index file. If Pinecone, upsert all vectors now so the index is ready.
	4.	Test the endpoints from a tool like cURL or Postman.
	5.	Deploy the front-end (upload files to S3 or push to Amplify).
	6.	Test end-to-end from the front-end URL.
	7.	Enable monitoring/alarm.
	8.	Document how to add new data and redeploy if needed.

By following this hosting plan, we ensure the MVP is not just a prototype running on a local machine, but a real web service accessible to users, with the scalability and reliability that AWS infrastructure provides.

8. Additional Notes
	•	Foundation for Future Development: This PRD is written to guide the MVP implementation, but we intentionally structure it in a way that it can lead into a more detailed technical design and architecture document. For each feature and component described, further design considerations (such as class structure, specific API contract details, and component diagrams) can be elaborated in a follow-up Technical Design Document (TDD). The PRD focuses on what needs to be built; the next document can focus on how exactly it will be built in terms of software architecture.
	•	Flexibility and Iteration: While we have defined clear acceptance criteria and deliverables, the team should remain agile. If during development we discover a certain approach doesn’t work well (for example, a chosen model isn’t accurate enough, or the multi-step agent is too complex), we can iterate on the plan. The PRD allows for some flexibility – e.g., switching out the vector DB or adjusting the number of results – as long as the core objectives are met. Any significant changes should be communicated and if needed, updated in this document for alignment.
	•	Out of Scope (for MVP): It’s worth noting what we are not doing in this MVP:
	•	We are not building a user management or authentication system. We assume the tool can be used without login for now (or with a simple API key if needed for protection).
	•	We are not doing complex parsing of resumes (we assume structured or plain text input; no PDF parsing or NLP entity extraction beyond embeddings).
	•	We are not integrating with external systems like LinkedIn or ATS (Applicant Tracking Systems) at this stage. Data is provided to the system manually or via simple upload.
	•	No advanced filtering or boolean logic beyond the basic refine agent. For instance, we’re not building a full query language or supporting very fine-grained filters in the UI (like filter by years experience via UI fields). Many of those could be future enhancements.
	•	Metrics on bias/fairness or ethical AI considerations in hiring – extremely important in real deployment – are not addressed in MVP. We should note that as a concern for future (ensuring the AI doesn’t unfairly rank candidates due to irrelevant factors), possibly by keeping an eye on what features the embedding picks up, but a full bias audit is future work.
	•	Assumptions: We assume the availability of certain resources:
	•	We have access to OpenAI API (with sufficient quota).
	•	We have or can create a dataset of resumes and jobs for testing. If not actual data, we’ll fabricate some realistic dummy data.
	•	Users (recruiters/job seekers) will provide input in English (multilingual matching could be a future expansion).
	•	The open-source models we use (SentenceTransformer, etc.) can be legally used for this purpose (they usually can, being MIT or similar license).
	•	Risks and Mitigations: A summary from previous sections:
	•	Data Privacy: Resume data is sensitive. In MVP, handle carefully (no broad logging, secure storage). In future, encryption and strict access control would be needed.
	•	Reliability: Relying on external APIs (OpenAI) means dependency risk. If OpenAI is down, our service is partially down. Mitigation in MVP might be minimal (we can’t control OpenAI). Future might involve a fallback model or caching.
	•	User Acceptance: Recruiters might be skeptical of AI suggestions. That’s why explanations are crucial – to justify and increase trust. We should gather feedback and be ready to adjust how we generate explanations to make them more aligned with what recruiters expect (maybe more structured like “Match Score: 9/10. Criteria matched: …”).
	•	Scalability of Embeddings: If the customer wants to load 100k resumes, Pinecone or OpenSearch will be needed; FAISS in Lambda won’t scale. Our design accounts for switching to those solutions, but MVP will likely only be tested with a small dataset.
	•	Future Enhancements: Although out of scope for MVP, it’s useful to note what features could be added later, to ensure our MVP design won’t block them:
	•	Enhanced Filtering: e.g., filter by location or salary range via UI fields.
	•	Profile/Job Editing: allow users to input their data through a form instead of pasting resume text, which then the system uses.
	•	Learning from Feedback: users could thumbs-up/thumbs-down suggestions, and we could learn to adjust (reinforcement learning or just fine-tune ranking).
	•	Use of Proprietary Models: maybe integrate with domain-specific models or fine-tune the embedding model on recruitment data to improve accuracy.
	•	Multi-Language Support: allow resumes and jobs in other languages by using multilingual embeddings.
	•	Bias Mitigation: ensure the matching is fair (e.g., maybe omit certain sensitive attributes from consideration, which could be a feature).
	•	Integration: allow exporting the results or integrating with an ATS, LinkedIn, etc., for direct outreach.
	•	Conclusion: This PRD outlines a clear path to build a minimum viable product of an AI-driven candidate matcher. By focusing on the core matching functionality with state-of-the-art AI techniques (embeddings + LLMs), and ensuring the solution is deployable on robust infrastructure, we aim to deliver immediate value to recruiters and job seekers. The detailed acceptance criteria and test plans will guide the development team in implementation and verification. Upon completion of this MVP, we will have a solid foundation to expand the product with more features and to refine the matching algorithms based on real user feedback, ultimately moving closer to a fully featured AI recruiting assistant.