from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import analyze

app = FastAPI(
    title="PoseMind API",
    description="AI 动作分析系统 API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js 开发服务器地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含分析路由
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])

@app.get("/")
async def root():
    return {"message": "欢迎使用 PoseMind API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 