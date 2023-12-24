import { HeaderButton, PopUpContainer } from '@/components/PopUpContainer/PopUpContainer'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import axios from "axios";
import { Spinner } from '@/components/Spinner'
import { Button } from '@/components/ui/button';
import { toast } from '@/components/ui/use-toast';
import { useRef } from 'react';

export const AddPlatePopup = () => {
    const navigate = useNavigate()
    const inputRef = useRef<HTMLInputElement>(null);
    const { mutate, isPending } = useMutation({
        mutationFn: addPlate,
        onSuccess: (_, plate) => {
            toast({
                description: `Your plate number:${plate} was added.`,
                variant: "secondary",
                duration: 2000,
                className: "py-4 bg-[#00afbf]",
            });
            navigate(-1)
        }
    })
    const handleSubmit = () => {
        if(inputRef.current?.value.trim().length==0) return;
        inputRef.current&& mutate(inputRef.current.value);

    }
    const handleXClick = () => {
        navigate(-1)
    }
    return (
        <PopUpContainer
            headerButton={HeaderButton.close}
            headerFunction={handleXClick}
            show showLogo className=' justify-start items-start pt-10 px-10' dialogClassName={`bg-black border border-[#00afbf] `}>
            {isPending ? <div className='w-full h-[180px] p-8'>
                <Spinner />
            </div> :
                <div className='flex flex-col justify-between h-[90%] items-start w-full'>
                    <div className='w-full'>
                        <label htmlFor="first_name" className="block font-bold  mb-2 text-xl text-gray-900 text-[#00afbf]">Add a Plate :</label>
                        <input ref={inputRef} type="text" id="first_name" className="bg-gray-50 border  border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="Plate Number" required />
                    </div>
                    <Button variant={"default"} className="bg-[#00afbf] rounded-xl w-full transition-all text-primary hover:bg-transparent border-2 border-[#00afbf] hover:text-[#00afbf]"  onClick={handleSubmit} >Add plate</Button>
                </div>}
        </PopUpContainer>
    )
}
const addPlate = async (licensePlate: string) => {
    try {
        const res = await axios.post('http://127.0.0.1:5000//add_plate', { license_plate: licensePlate });
        return res.data;
    } catch (err) {
        console.log(err);
        return null;
    }
}