# SupplyChainReact

This project is for the exam of the course Blockchain & Cryptocurrencies from the MSc in Artificial Intelligence at the University of Bologna.
The following is a Decentralized App for the supply chain to track the journey of raw materials from the farmers to the consumers. 

With this app we can track the origin of raw materials, how they are processed and their output from the manufacturers, and their journey through distributors and retailers.

## Dependencies
Contracts:
 - Truffle: ^5.5.13
 - Solidity compiler: ^0.8.13

Client: 
 - @material-ui/core: ^4.11.4 for react template components
 - @material-ui/icons: ^4.11.2 for icons
 - antd: ^4.20.4 for explorer (to show products and raw materials in table) 
 - clsx: ^1.1.1
 - express: ^4.17.1
 - ganache-cli: ^6.12.2 for testing the interactions of accounts
 - ganache-core: ^2.13.2
 - react: ^17.0.2  for client interface
 - react-dom: ^17.0.2 
 - react-router-dom: ^5.2.0
 - react-scripts: ^3.2.0
 - web3: ^1.6.1 for interactions with metamask


## Usage

  - Clone the repository
  - Open Ganache UI (has issues with storing mappings) or Ganache-cli with `ganache-cli -p 7545 -i 5777`
  - `truffle test` to check tests of smart contract
  - `truffle compile` to compile contracts
  - `truffle migrate ganache --reset` to deploy smart contract on ganache network (localhost:7545)
  - Run the client:

    ```
    cd client
    npm -i       //to install dependencies
    npm start    //to start server
    ```
   
 After starting the app, you need to connect Metamask - use the mnemonic provided in Ganache UI or CLI and provide a common password
 
 Remember to set the Metamask network on a custom Test network with parameters https:127.0.0.1:7545 with chainID 1337 and any symbol you want.
 
 To import other accounts, Go on Ganache UI -> Accounts -> Copy the private key. Then on Metamask extension click on the account -> Import account and paste 
 the private key. Remember to connect the account to the site and set the correct network.

 ## Activity

 ![Activity](https://github.com/FrancescoFarinola/MSc_Artificial_Intelligence/blob/main/Blockchain%20%26%20Cryptocurrencies/images/activity.png)

 ## UI

![UI](https://github.com/FrancescoFarinola/MSc_Artificial_Intelligence/blob/main/Blockchain%20%26%20Cryptocurrencies/images/ui.png)

