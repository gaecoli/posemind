import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from services.pose_analyzer import PoseAnalyzer
import tempfile

router = APIRouter()
analyzer = PoseAnalyzer()

@router.post("/pushup")
async def analyze_pushup(video: UploadFile = File(...)):
    try:
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 分析视频
        result, scores = analyzer.analyze_pushup(temp_file_path)

        # 删除临时文件
        os.unlink(temp_file_path)

        return JSONResponse({
            "message": result,
            "scores": scores
        })

    except Exception as e:
        return JSONResponse({
            "message": f"分析过程中出现错误: {str(e)}"
        }, status_code=500) 