// Inspect Fast DDS's actual parsed profile before any participant is created.
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastrtps/utils/IPLocator.h>

#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
    using eprosima::fastdds::dds::DomainParticipantFactory;
    using eprosima::fastdds::rtps::UDPv4TransportDescriptor;
    using eprosima::fastrtps::rtps::IPLocator;

    if (argc != 2) {
        return 2;
    }
    auto* factory = DomainParticipantFactory::get_instance();
    if (factory->load_XML_profiles_file(argv[1]) !=
            eprosima::fastrtps::types::ReturnCode_t::RETCODE_OK) {
        return 3;
    }
    const auto qos = factory->get_default_participant_qos();
    const auto& transport = qos.transport();
    if (transport.use_builtin_transports || transport.user_transports.size() != 1) {
        return 4;
    }
    const auto udp = std::dynamic_pointer_cast<UDPv4TransportDescriptor>(
        transport.user_transports.front());
    if (!udp || udp->interfaceWhiteList.size() != 1 ||
            udp->interfaceWhiteList.front() != "127.0.0.1") {
        return 5;
    }
    const auto& builtin = qos.wire_protocol().builtin;
    if (builtin.initialPeersList.size() != 1 ||
            builtin.metatrafficUnicastLocatorList.size() != 1 ||
            !builtin.metatrafficMulticastLocatorList.empty()) {
        return 6;
    }
    if (IPLocator::toIPv4string(*builtin.initialPeersList.begin()) != "127.0.0.1" ||
            IPLocator::toIPv4string(*builtin.metatrafficUnicastLocatorList.begin()) != "127.0.0.1") {
        return 7;
    }
    std::cout << "VERIFIED_UDP4_LOOPBACK_ONLY" << std::endl;
    return 0;
}
