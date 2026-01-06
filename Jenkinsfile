// Jenkinsfile
// Author : Trần Quý Đạt
// Email  : tranquydat.work@gmail.com
// Project: Vietnamese Document OCR Serving Model - MLOps Pipeline
//
// CI/CD Pipeline triggered by a deployment cron or GitHub webhook.
// Stages: build training image → push to DockerHub → apply K8s manifests

pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        timestamps()
    }

    environment {
        TRAINING_IMAGE   = 'tranquydat/vn-doc-ocr-training'
        REGISTRYCREDENTIAL = 'dockerhub'
    }

    stages {
        stage('Build Training Image') {
            steps {
                script {
                    echo 'Building OCR training Docker image...'
                    def dockerImage = docker.build(
                        "${TRAINING_IMAGE}:${BUILD_NUMBER}",
                        "distributed_training/."
                    )
                    echo 'Pushing image to DockerHub...'
                    docker.withRegistry('', REGISTRYCREDENTIAL) {
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Model Optimization - Export ONNX') {
            steps {
                echo 'Exporting PaddleOCR inference models to ONNX via paddle2onnx...'
                sh '''
                    pip install -q paddle2onnx==1.3.0
                    python distributed_training/export_onnx.py --all
                '''
            }
        }

        stage('Model Testing') {
            steps {
                echo 'Running model unit tests...'
                sh 'pip install -q pytest && pytest tests/ -v --tb=short'
            }
        }

        stage('Upload Model to MinIO') {
            steps {
                echo 'Ingesting ONNX model artefacts to MinIO (S3)...'
                withCredentials([
                    usernamePassword(
                        credentialsId: 'minio-creds',
                        usernameVariable: 'MINIO_ACCESS_KEY',
                        passwordVariable: 'MINIO_SECRET_KEY'
                    )
                ]) {
                    sh 'python api/upload_model_to_minio.py'
                }
            }
        }

        stage('Deploy to KServe') {
            steps {
                echo 'Applying KServe InferenceService manifest...'
                sh 'kubectl apply -f deployments/triton-isvc.yaml'
                sh 'kubectl apply -f deployments/triton-servingruntime.yaml'
            }
        }
    }

    post {
        success {
            echo "Pipeline succeeded — build #${BUILD_NUMBER} deployed."
        }
        failure {
            mail to: 'tranquydat.work@gmail.com',
                 subject: "[FAILED] OCR Pipeline Build #${BUILD_NUMBER}",
                 body: "Check Jenkins: ${BUILD_URL}"
        }
    }
}
                    echo 'Pushing image to dockerhub..'
                    docker.withRegistry( '', registryCredential ) {
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }
        // stage('Deploy') {
        //     steps {
        //         echo 'Deploying models..'
        //         echo 'Running a script to trigger pull and start a docker container'
        //     }
        // }
    }
}
