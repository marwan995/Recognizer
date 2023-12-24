import { Particles } from 'react-tsparticles'
import { loadFull } from 'tsparticles'
import { useCallback } from 'react'
const ParticlesContainer = () => {

    const particlesInit = useCallback(async (engine: any) => {
        await loadFull(engine )
    }, [])
    const particlesLoaded = useCallback(async () => { }, []);
    return (<Particles className='w-[88%] max-md:w-[88%] h-[95%] absolute translate-z-0 z-[-1] ' id='tsparticles' init={particlesInit} loaded={particlesLoaded}
        options={{
            fullScreen: { enable: false },
            background: {
                color: {
                    value: ''
                },
            },
            fpsLimit: 60,
            interactivity: {
                events: {
                    onClick: {
                        enable: false,
                        mode: 'push'
                    },
                    onHover: {
                        enable: true,
                        mode: 'repulse'
                    },
                    resize: true,
                },
                modes: {
                    push: {
                        quantity: 90
                    },
                    repulse: {
                        distance: 150,
                        duration: 0.4,
                    }
                }

            },
            particles: {
                color: {
                    // value: '#e68e2e',
                    value: '#00afbf',

                },
                links: {
                    // color: '#f5d393',
                    color: '#00afbf',
                    distance: 150,
                    enable: true,
                    opacity: 0.5,
                    width: 1
                },
                collisions: {
                    enable: true
                },
                move: {
                    direction: 'none',
                    enable: true,
                    outModes: {
                        default: 'bounce'
                    },
                    random: false,
                    speed: 1,
                    straight: false
                },
                number: {
                    density: {
                        enable: true,
                        area:500,
                    },
                    value:60
                },
                opacity:{value:0.5},
                shape:{type:'circle'},
                size:{
                    value:{min:1,max:5},
                }


            },
                detectRetina:true
        }}
    />
    )
}
export default ParticlesContainer