import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from services.pose_analyzer import PoseAnalyzer
import tempfile
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
analyzer = PoseAnalyzer()

@router.post("/pushup")
async def analyze_pushup(video: UploadFile = File(...)):
    try:
        logger.info(f"收到视频文件: {video.filename}, 类型: {video.content_type}")
        
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"视频已保存到临时文件: {temp_file_path}")

        # 分析视频
        logger.info("开始分析视频...")
        result, scores = analyzer.analyze_pushup(temp_file_path)
        logger.info(f"分析完成: {result}")
        logger.info(f"得分: {scores}")

        # 删除临时文件
        os.unlink(temp_file_path)
        logger.info("临时文件已删除")

        return JSONResponse({
            "message": result,
            "scores": scores
        })
    except Exception as e:
        logger.error(f"分析过程中出现错误: {str(e)}")
        return JSONResponse({
            "message": f"分析过程中出现错误: {str(e)}",
            "scores": []
        }, status_code=500)

@router.post("/shooting")
async def analyze_shooting(video: UploadFile = File(...)):
    try:
        logger.info(f"收到视频文件: {video.filename}, 类型: {video.content_type}")
        
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"视频已保存到临时文件: {temp_file_path}")

        # 分析视频
        logger.info("开始分析视频...")
        result, scores = analyzer.analyze_shooting(temp_file_path)
        logger.info(f"分析完成: {result}")
        logger.info(f"得分: {scores}")

        # 删除临时文件
        os.unlink(temp_file_path)
        logger.info("临时文件已删除")

        return JSONResponse({
            "message": result,
            "scores": scores
        })
    except Exception as e:
        logger.error(f"分析过程中出现错误: {str(e)}")
        return JSONResponse({
            "message": f"分析过程中出现错误: {str(e)}",
            "scores": []
        }, status_code=500) 