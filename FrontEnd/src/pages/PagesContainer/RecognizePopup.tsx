import { HeaderButton, PopUpContainer } from '@/components/PopUpContainer/PopUpContainer'
import { cn } from '@/lib/utils'
import { useMutation } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import axios from "axios";
import { Spinner } from '@/components/Spinner'

export const RecognizePopup = () => {
    const location = useLocation()
    const navigate = useNavigate()
    const [canPass, setCanPass] = useState(false)
    const [license, setLicense] = useState("")
    const [predictImg, setPredictImg] = useState("");

    const img = location.state.file;
    const { mutate, isPending } = useMutation({
        mutationFn: getPrediction,
        onSuccess: (data) => {
            console.log(data)
            setCanPass(data.canPass)
            setLicense(data.license)
            setPredictImg('http://127.0.0.1:5000/images/'+data.plateImg);
        }
    })
    useEffect(() => {
        const formData = new FormData();
        formData.set('file', img)
        mutate(formData);
    }, [])
    const handleXClick = () => {
        navigate(-1)
    }
    return (
        <PopUpContainer
            headerButton={HeaderButton.close}
            headerFunction={handleXClick}
            show showLogo className=' justify-start items-start pt-10 px-10' dialogClassName={`bg-black border  ${canPass ? 'border-[#00afbf]' : 'border-danger'} `}>
            {isPending ? <div className='w-full h-[180px] p-8'>
                <Spinner />
            </div> : <div className='flex flex-col justify-start items-start'>
                <div className={cn(' border md:ml-5 max-h-[50vh] rounded-xl overflow-scroll overflow-x-hidden border-opacity-25 scroll-0 scrollbar-thin', canPass ? 'border-[#00afbf]' : 'border-danger')}>
                    <img src={URL.createObjectURL(img)} className='object-cover  my-2  border-b-8 border-[#00afbf]  border-solid' />
                    {predictImg && <img src={predictImg} className='object-cover w-full' />}
                </div>
                {canPass ? <h3 className=' mx-auto mt-10 font-bold text-xl text-[#00afbf]'> [{license}] CAN GO  （＾∀＾●）ﾉｼ  </h3> : <>
                    <h3 className='mx-auto mt-10 font-bold text-xl text-danger'> [{license}] STOP ￣へ￣  </h3>
                    <span className=' mx-auto font-bold text-xl text-danger'>  YOU CAN'T PASS </span></>}
            </div>}
        </PopUpContainer>
    )
}
const getPrediction = async (form: FormData) => {
    try {
        const res = await axios.post('http://127.0.0.1:5000/predict', form);
        console.log(res.data)
        return res.data;
    } catch (err) {
        console.log(err);
        return null;
    }
}