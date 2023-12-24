
import { Link, useLocation, useNavigate } from "react-router-dom";
import img from './5738467.png'
import logo from './recognizer-high-resolution-logo-transparent.png'
import { Button } from "@/components/ui/button";
import { useRef } from "react";
export function PagesContainer() {
  const location = useLocation();
  const navigate = useNavigate();
  const fileInput = useRef<HTMLInputElement>(null);
  const handleUploadImage = () => {
    fileInput.current?.click()

  }
  const handleFileChange = () => {
    const filesTemp = fileInput.current!.files;
    if (filesTemp && filesTemp.length > 0) {
      const selectedFile = filesTemp[0];
      console.log(selectedFile)
      navigate('/recognize', { state: { previousLocation: location, file: selectedFile } })

    }
  }

  return (
    <div className="w-[100vw] min-h-[100vh] p-5 bg-black text-primary justify-between flex flex-row max-md:flex-col gap-10   ">

      <div className=" flex flex-col gap w-[48vw] max-md:w-[90vw]">
        <img src={logo} alt="" className="w-[150px] h-[100px]" />

        <h1 className="font-bold mt-[10vh] text-[50px] text-[#00afbf]">Recognizer</h1>
        <p className=" text-primary text-bold ">Our Intelligent License Plate Recognition and Validation System is an advanced image processing project designed to automate and enhance the process of license plate recognition and validation. This innovative solution combines cutting-edge image processing techniques.</p>
        <div className="flex flex-row gap-5 mt-[5vh] max-md:flex-col">

          <Button variant={"default"} className="bg-[#00afbf] rounded-xl text-primary hover:bg-transparent border-2 border-[#00afbf] hover:text-[#00afbf]"  onClick={handleUploadImage} >Recognize an image</Button>
          <Link to={"/add_plate"} state={ { previousLocation:location}} >
            <Button className="bg-transparent rounded-xl hover:text-primary hover:bg-[#00afbf] w-full border-2 border-[#00afbf] text-[#00afbf] px-5" >add a plate</Button>
          </Link>
        </div>
      </div>
      <img src={img} alt="" className=" w-[50vw] aspect-video  max-md:w-[80vw] max-md:h-[50vh] max-md:hidden" />
      <input type="file" className="hidden" onChange={handleFileChange} ref={fileInput} accept={"image/*"} multiple={false} />
    </div >
  );
}