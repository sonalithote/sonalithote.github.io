---
layout: projects
# collectionpage: posts
permalink: /projects/
# title: Projects
categories:
- General
- External sources
feature_image: "https://picsum.photos/2560/600?image=872"
feature_text: Welcome to the Projects
---

## Project 1

### React Application with Docker 

[Git link](https://github.com/sonalithote/docker-react/tree/master "Git link")


This project repository contains a simple React application that is containerized using Docker. This allows for easy deployment and consistent environments across different platforms.

* Prerequisites
Before you begin, ensure that you have Docker installed on your machine. You can download Docker from https://www.docker.com/get-started.

* Build and Run
Follow these steps to build and run the React application using Docker:

1. Clone this repository to your local machine:
git clone https://github.com/your-username/react-docker-app.git
cd react-docker-app
2. Build the Docker image:
docker build -t react-docker-app .
3. Run the Docker container:
docker run -p 3000:3000 react-docker-app
This will start the React application and expose it on http://localhost:3000.

* Development
If you want to make changes to the React application, you can follow these steps:
1. Install dependencies:
npm install
2. Start the development server:
npm start
This will run the React application in development mode on http://localhost:3000.

* Docker Configuration
The Dockerfile in this repository includes all the necessary configurations to package the React application into a Docker image. You can customize this file according to your project requirements.

* License
This React application is open-source and available under the MIT License.


## Project 2

### Securing REST APIs with Role-Based OAuth2 Implementation 

[Git link](https://github.com/sonalithote/rolebasedoauthspringboot/tree/master "Git link")


This project in Git repository demonstrates the implementation of securing REST APIs using role-based OAuth2. In this project, we will create custom roles, namely ADMIN and USER, and utilize the @secured annotation provided by Spring Security to secure controller methods based on roles.

* Project Overview
Custom Roles: We will define two roles, ADMIN and USER, and assign access permissions accordingly.
Spring Security: Utilizing Spring Security, we will control access to our API endpoints based on the user's role.
MySQL Database: User details, credentials, and associated roles will be stored in a MySQL database. Spring Data will be employed for efficient database operations.
Spring Boot: The project will leverage Spring Boot for streamlined configuration and ease of development.

* Implementation Details
1. Role-Based Access Control:
Use @secured annotation to enforce role-based access control on controller methods.
Grant access to certain endpoints exclusively to users with the ADMIN role.
2. Database Operations:
Store user details, credentials, and roles in a MySQL database.
Leverage Spring Data to handle database operations efficiently.
3. Testing with Postman:
Postman will be employed for performing CRUD operations and testing all APIs.
4. Token Authentication:
Implement JwtTokenStore to translate access tokens to and from authentications.

* Usage
To run and test the project:
1. Clone the repository.
2. Configure your MySQL database settings in the application properties.
3. Run the Spring Boot application.
4. Use Postman to interact with the API endpoints and perform CRUD operations.