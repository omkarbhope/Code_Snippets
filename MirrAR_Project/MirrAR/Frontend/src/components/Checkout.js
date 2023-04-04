import React, { useState }  from 'react'
import CartIcon from './CartIcon'
import Navbar from './Navbar'

import lips1 from '../images/pac1.jpg'



export default function Checkout(props) {

    let content = [];

    let checkoutlistfull = []


    function convertSessoinCheckoutListIntoHash() {

        var lengthCheckout = JSON.parse(sessionStorage.getItem("checkoutlist")).length
        if(lengthCheckout > 0) {
            for(var i = 0; i < lengthCheckout; i++) {
                checkoutlistfull.push(JSON.parse(JSON.parse(sessionStorage.getItem("checkoutlist"))[i]))
            }
            
        }
        console.log(checkoutlistfull) 
        
    }


    convertSessoinCheckoutListIntoHash()



    const [list, updateList] = useState(checkoutlistfull);
    let [counter, setCounter] = useState(list.length)



   function listItemsToStringyfy(ls)  {
        var newlisttoStringyfy = []
        for(var i = 0; i < ls.length ; i++ ) {
            newlisttoStringyfy.push(JSON.stringify(ls[i]))
        }
        return newlisttoStringyfy
    }


    const revomeFromContent = (id) => {
        
        var newlist = list.splice(id, 1)
        updateList(list);
        
        sessionStorage.setItem("checkoutlist", JSON.stringify(listItemsToStringyfy(list)))
        setCounter(list.length)
        console.log(list.length)
      
    }



    const getContent = () => {


        for(let i = 0; i <counter; i++) { 

            content.push(

            <div className="checkout__div">
                
                <div className="checkout__div__img">
                    <img src={list[i].imageProduct} alt=""/>
                </div>

                <div className="checkout__div__info">
                    <div className="checkout__div__header">
                        <h2>{list[i].descProduct}</h2>
                        <span className='deletefrom__checkout' onClick={() => revomeFromContent(i)}>X</span>
                    </div>
                    <div className="checkout__div__footer">
                        
                        <h4>
                            <span>{list[i].priceProduct}</span>
                        </h4>
                    </div>
                </div>
                    
            </div>

            )


        }

        return content;

    }


  return (
   
    <>
        <Navbar/>
        <section className="checkout__section">
            <div className="checkout__divs">
                {getContent()}
            </div>
        </section>
        <CartIcon countCart = {counter}/>
    </>



  )
}
